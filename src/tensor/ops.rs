//! Tensor ops and related functionality.

use std::cell::Ref;

use pliron::{
    builtin::op_interfaces::{
        AllOperandsOfType, AllResultsOfType, NOpdsInterface, NRegionsInterface, NResultsInterface,
        OneRegionInterface, OneResultInterface, OperandSegmentInterface,
        SingleBlockRegionInterface,
    },
    combine::{
        Parser,
        parser::char::{self, spaces},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    identifier::Identifier,
    irbuild::{
        inserter::{BlockInsertionPoint, IRInserter, Inserter, OpInsertionPoint},
        listener::DummyListener,
    },
    irfmt::{
        parsers::{delimited_list_parser, process_parsed_ssa_defs, spaced, ssa_opd_parser},
        printers::iter_with_sep,
    },
    location::Location,
    op::{Op, OpObj},
    operation::Operation,
    parsable::{self, IntoParseResult, Parsable},
    printable::{self, Printable},
    result::Result,
    r#type::{TypeObj, TypePtr, Typed, type_cast},
    value::Value,
    verify_err, verify_error,
};

use pliron_common_dialects::{cf::op_interfaces::YieldingRegion, index::types::IndexType};

use crate::memref::{
    op_interfaces::{CompatibleShapesOp, GenerateOpInterface},
    ops::YieldOp,
    type_interfaces::{MultiDimensionalType, ShapedType},
};

use super::{op_interfaces::ElementWiseBinaryTensorOpInterface, types::RankedTensorType};

/// Op to generate a tensor by applying a function to generate the value at each index.
/// See MLIR's [GenerateOp](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgenerate-tensorgenerateop).
///
/// ### Operands(s)
/// | operand | description |
/// |-----|-------|
/// | `dynamic_dimensions` | One [Index](IndexType) operand per dynamic dimension, to indicate the extent of that dimension |
///
/// ### Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The generated tensor of the specified type. |
///
/// ### Regions
///   - A single region containing the body that computes the values of the tensor.
///   The region takes as many arguments as the rank of the result tensor type,
///   each representing an index along the corresponding dimension. The body should
///   yield a single value that matches the element type of the tensor.
#[pliron_op(
    name = "tensor.generate",
    format = "operands(CharSpace(`,`)) ` : ` type($0) region($0)",
    interfaces = [
        SingleBlockRegionInterface,
        OneRegionInterface,
        NRegionsInterface<1>,
        OneResultInterface,
        NResultsInterface<1>,
        YieldingRegion<YieldOp>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<IndexType>,
    ],
)]
pub struct GenerateOp;

#[derive(thiserror::Error, Debug)]
pub enum GenerateOpVerifyErr {
    #[error(
        "GenerateOp number of operands {expected} does not match number of dynamic dimensions {got}"
    )]
    NumOperandsMismatch { expected: usize, got: usize },
}

impl Verify for GenerateOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let result_shape = self.get_generated_shape(ctx);
        let num_dynamic_dims = result_shape.num_dynamic_dimensions();

        let dynamic_dim_operands = self
            .get_operation()
            .deref(ctx)
            .operands()
            .collect::<Vec<_>>();
        let num_operands = dynamic_dim_operands.len();
        if num_operands != num_dynamic_dims {
            return verify_err!(
                loc,
                GenerateOpVerifyErr::NumOperandsMismatch {
                    expected: num_dynamic_dims,
                    got: num_operands
                }
            );
        }
        Ok(())
    }
}

impl GenerateOpInterface for GenerateOp {
    fn get_generated_shape<'a>(&'a self, ctx: &'a Context) -> Ref<'a, dyn ShapedType> {
        let result_ty = self.result_type(ctx).deref(ctx);
        Ref::map(result_ty, |result_ty| {
            type_cast::<dyn ShapedType>(&**result_ty)
                .expect("The result type must be a shaped type")
        })
    }
}

impl GenerateOp {
    /// Creates a new dynamically sized tensor.
    /// The `body_builder` function is called to populate the body of the region.
    /// It is provided with, as arguments, the current index values and an inserter
    /// (set to the end of the entry block). It must return the value yielded at that index.
    /// A [YieldOp] is automatically added at end of the body, taking this value as operand.
    pub fn new<State>(
        ctx: &mut Context,
        dynamic_dimensions: Vec<Value>,
        result_type: TypePtr<RankedTensorType>,
        body_builder: fn(
            ctx: &mut Context,
            state: State,
            inserter: &mut IRInserter<DummyListener>,
            indices: Vec<Value>,
        ) -> Value,
        body_builder_state: State,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type.into()],
            dynamic_dimensions,
            vec![],
            1,
        );
        let opop = GenerateOp { op };
        let rank = result_type.deref(ctx).rank();

        // Create the initializer region.
        let index_ty = IndexType::get(ctx);
        let region = opop.get_region(ctx);
        let op_inserter = &mut IRInserter::default();
        let entry_block = op_inserter.create_block(
            ctx,
            BlockInsertionPoint::AtRegionStart(region),
            Some("entry".try_into().unwrap()),
            vec![index_ty.into(); rank],
        );
        // Build the body.
        let indices = entry_block.deref(ctx).arguments().collect();
        let yield_value = body_builder(ctx, body_builder_state, op_inserter, indices);
        let yield_op = YieldOp::new(ctx, yield_value);
        op_inserter.set_insertion_point(OpInsertionPoint::AtBlockEnd(opop.get_exit(ctx)));
        op_inserter.append_op(ctx, yield_op);

        opop
    }

    /// Get the dynamic dimension operands of this op.
    pub fn dynamic_dimensions(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

/// Extract an element from a tensor at the given indices.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `tensor` | The tensor to extract from. |
/// | `indices` | One [Index](IndexType) operand per dimension, indicating the index to extract along that dimension. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The extracted element, with the same type as the element type of the operand tensor. |
#[pliron_op(
    name = "tensor.extract",
    interfaces = [OneResultInterface, NResultsInterface<1>, OperandSegmentInterface]
)]
pub struct ExtractOp;

impl Printable for ExtractOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let tensor = self.get_tensor_operand(ctx);
        let indices = self.get_index_operands(ctx);
        write!(
            f,
            "{} {}[{}] : {}",
            Self::get_opid_static(),
            tensor.disp(ctx),
            iter_with_sep(
                indices.iter(),
                pliron::printable::ListSeparator::CharSpace(',')
            )
            .disp(ctx),
            self.result_type(ctx).disp(ctx)
        )
    }
}

impl Parsable for ExtractOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> parsable::ParseResult<'a, Self::Parsed> {
        let (memref, indices, res_ty) = (
            ssa_opd_parser().skip(spaces()),
            delimited_list_parser('[', ']', ',', ssa_opd_parser()),
            spaced(char::string(":")).with(Ptr::<TypeObj>::parser(())),
        );

        let ((memref, indices, res_ty), _) = (memref, indices, res_ty)
            .parse_stream(state_stream)
            .into_result()?;

        let op = ExtractOp::new(state_stream.state.ctx, res_ty, memref, indices);

        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

impl ExtractOp {
    /// Create a new ExtractOp with the given operand and result type.
    pub fn new(
        ctx: &mut Context,
        res_ty: Ptr<TypeObj>,
        tensor: Value,
        indices: Vec<Value>,
    ) -> Self {
        let (operands, operand_segments) = Self::compute_segment_sizes(vec![vec![tensor], indices]);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![res_ty],
            operands,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_operand_segment_sizes(ctx, operand_segments);
        op
    }

    /// Get the operand representing the tensor to extract from.
    pub fn get_tensor_operand(&self, ctx: &Context) -> Value {
        self.get_segment(ctx, 0)[0]
    }

    /// Get the operands representing the indices to extract at.
    pub fn get_index_operands(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 1)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExtractOpVerifyErr {
    #[error("ExtractOp must have at least one operand")]
    NoOperands,
    #[error("The first operand of ExtractOp must be a RankedTensorType")]
    FirstOperandNotTensor,
    #[error("The result type of ExtractOp must match the element type of the operand tensor")]
    ResultTypeMismatch,
    #[error("The number of operands must match the rank of the operand tensor")]
    NumOperandsMismatch { expected: usize, got: usize },
    #[error("All operands except the first one must be of IndexType")]
    NonIndexOperand { index: usize },
}

impl Verify for ExtractOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let op_ref = self.get_operation().deref(ctx);
        let mut operand_tys = op_ref.operands().map(|opd| opd.get_type(ctx));

        let Some(tensor_operand_ty) = operand_tys.next() else {
            return verify_err!(loc, ExtractOpVerifyErr::NoOperands);
        };

        let tensor_operand_ty_ref = tensor_operand_ty.deref(ctx);
        let ranked_tensor_ty = tensor_operand_ty_ref
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), ExtractOpVerifyErr::FirstOperandNotTensor))?;
        let element_ty = ranked_tensor_ty.element_type();
        let result_ty = self.result_type(ctx);
        if result_ty != element_ty {
            return verify_err!(loc, ExtractOpVerifyErr::ResultTypeMismatch);
        }
        let expected_num_indices = ranked_tensor_ty.rank();
        let mut num_indices = 0;
        for (i, index_ty) in operand_tys.enumerate() {
            let index_ty_ref = index_ty.deref(ctx);
            if !index_ty_ref.is::<IndexType>() {
                return verify_err!(loc, ExtractOpVerifyErr::NonIndexOperand { index: i });
            }
            num_indices += 1;
        }
        if num_indices != expected_num_indices {
            return verify_err!(
                loc,
                ExtractOpVerifyErr::NumOperandsMismatch {
                    expected: expected_num_indices,
                    got: num_indices
                }
            );
        }
        Ok(())
    }
}

/// Add two tensors.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lhs` | The left-hand side tensor. |
/// | `rhs` | The right-hand side tensor. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The resulting tensor, with same shape as the operands. |
#[pliron_op(
    name = "tensor.add",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        OneResultInterface,
        ElementWiseBinaryTensorOpInterface,
        NResultsInterface<1>,
        NOpdsInterface<2>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<RankedTensorType>,
        CompatibleShapesOp<RankedTensorType>,
    ],
    verifier = "succ"
)]
pub struct AddOp;

impl AddOp {
    /// Create a new AddOp with the given operands and result type.
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![lhs.get_type(ctx)],
            vec![lhs, rhs],
            vec![],
            0,
        );
        Self { op }
    }
}

/// Subtract two tensors.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lhs` | The left-hand side tensor. |
/// | `rhs` | The right-hand side tensor. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The resulting tensor, with same shape as the operands. |
#[pliron_op(
    name = "tensor.sub",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        OneResultInterface,
        ElementWiseBinaryTensorOpInterface,
        NResultsInterface<1>,
        NOpdsInterface<2>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<RankedTensorType>,
        CompatibleShapesOp<RankedTensorType>,
    ],
    verifier = "succ"
)]
pub struct SubOp;

impl SubOp {
    /// Create a new SubOp with the given operands and result type.
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![lhs.get_type(ctx)],
            vec![lhs, rhs],
            vec![],
            0,
        );
        Self { op }
    }
}

/// Multiply two tensors.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lhs` | The left-hand side tensor. |
/// | `rhs` | The right-hand side tensor. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The resulting tensor, with same shape as the operands. |
#[pliron_op(
    name = "tensor.mul",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        OneResultInterface,
        ElementWiseBinaryTensorOpInterface,
        NResultsInterface<1>,
        NOpdsInterface<2>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<RankedTensorType>,
        CompatibleShapesOp<RankedTensorType>,
    ],
    verifier = "succ"
)]
pub struct MulOp;

impl MulOp {
    /// Create a new MulOp with the given operands and result type.
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![lhs.get_type(ctx)],
            vec![lhs, rhs],
            vec![],
            0,
        );
        Self { op }
    }
}

/// Divide two tensors.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lhs` | The left-hand side tensor. |
/// | `rhs` | The right-hand side tensor. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The resulting tensor, with same shape as the operands. |
#[pliron_op(
    name = "tensor.div",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        OneResultInterface,
        ElementWiseBinaryTensorOpInterface,
        NResultsInterface<1>,
        NOpdsInterface<2>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<RankedTensorType>,
        CompatibleShapesOp<RankedTensorType>,
    ],
    verifier = "succ"
)]
pub struct DivOp;

impl DivOp {
    /// Create a new DivOp with the given operands and result type.
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![lhs.get_type(ctx)],
            vec![lhs, rhs],
            vec![],
            0,
        );
        Self { op }
    }
}

/// Matrix multiplication of two 2D tensors.
/// `lhs` has shape [M, K], `rhs` has shape [K, N], and the result has shape [M, N].
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lhs` | Left-hand side 2D tensor of shape [M, K]. |
/// | `rhs` | Right-hand side 2D tensor of shape [K, N]. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The product tensor of shape [M, N]. |
#[pliron_op(
    name = "tensor.matmul",
    interfaces = [
        OneResultInterface,
        NResultsInterface<1>,
        NOpdsInterface<2>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<RankedTensorType>,
    ],
)]
pub struct MatMulOp;

impl Printable for MatMulOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let lhs = self.get_operation().deref(ctx).get_operand(0);
        let rhs = self.get_operation().deref(ctx).get_operand(1);
        write!(
            f,
            "{} {}, {} : {}",
            Self::get_opid_static(),
            lhs.disp(ctx),
            rhs.disp(ctx),
            self.result_type(ctx).disp(ctx)
        )
    }
}

impl Parsable for MatMulOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> parsable::ParseResult<'a, Self::Parsed> {
        let lhs = ssa_opd_parser().skip(spaced(char::string(",")));
        let rhs = ssa_opd_parser();
        let res_ty = spaced(char::string(":")).with(Ptr::<TypeObj>::parser(()));

        let ((lhs, rhs, res_ty), _) = (lhs, rhs, res_ty)
            .parse_stream(state_stream)
            .into_result()?;

        let op = MatMulOp::new_with_result_type(state_stream.state.ctx, lhs, rhs, res_ty);
        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MatMulOpVerifyErr {
    #[error("MatMulOp operands must be 2D ranked tensors")]
    OperandNot2DTensor,
    #[error("MatMulOp result must be a 2D ranked tensor")]
    ResultNot2DTensor,
    #[error("MatMulOp lhs inner dimension K ({lhs_k}) must match rhs outer dimension K ({rhs_k})")]
    InnerDimMismatch { lhs_k: usize, rhs_k: usize },
    #[error("MatMulOp result dimension {dim} (={result_d}) does not match expected {expected}")]
    ResultDimMismatch {
        dim: usize,
        result_d: usize,
        expected: usize,
    },
    #[error("MatMulOp operands and result must have the same element type")]
    ElementTypeMismatch,
}

impl Verify for MatMulOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        use crate::memref::type_interfaces::Dimension;
        let loc = self.loc(ctx);
        let op_ref = self.get_operation().deref(ctx);
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let result = op_ref.get_result(0);

        let lhs_ty_deref = lhs.get_type(ctx);
        let rhs_ty_deref = rhs.get_type(ctx);
        let result_ty_deref = result.get_type(ctx);

        let lhs_binding = lhs_ty_deref.deref(ctx);
        let lhs_ty = lhs_binding
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), MatMulOpVerifyErr::OperandNot2DTensor))?;
        let rhs_binding = rhs_ty_deref.deref(ctx);
        let rhs_ty = rhs_binding
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), MatMulOpVerifyErr::OperandNot2DTensor))?;
        let result_binding = result_ty_deref.deref(ctx);
        let result_ty = result_binding
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), MatMulOpVerifyErr::ResultNot2DTensor))?;

        if lhs_ty.rank() != 2 {
            return verify_err!(loc, MatMulOpVerifyErr::OperandNot2DTensor);
        }
        if rhs_ty.rank() != 2 {
            return verify_err!(loc, MatMulOpVerifyErr::OperandNot2DTensor);
        }
        if result_ty.rank() != 2 {
            return verify_err!(loc, MatMulOpVerifyErr::ResultNot2DTensor);
        }

        let elem_ty = lhs_ty.element_type();
        if rhs_ty.element_type() != elem_ty || result_ty.element_type() != elem_ty {
            return verify_err!(loc, MatMulOpVerifyErr::ElementTypeMismatch);
        }

        let lhs_shape = lhs_ty.shape();
        let rhs_shape = rhs_ty.shape();
        let result_shape = result_ty.shape();

        // K: lhs[1] must match rhs[0]
        if let (Dimension::Static(lhs_k), Dimension::Static(rhs_k)) = (&lhs_shape[1], &rhs_shape[0])
            && lhs_k != rhs_k
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::InnerDimMismatch {
                    lhs_k: *lhs_k,
                    rhs_k: *rhs_k
                }
            );
        }
        // M: lhs[0] must match result[0]
        if let (Dimension::Static(lhs_m), Dimension::Static(result_m)) =
            (&lhs_shape[0], &result_shape[0])
            && lhs_m != result_m
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::ResultDimMismatch {
                    dim: 0,
                    result_d: *result_m,
                    expected: *lhs_m
                }
            );
        }
        // N: rhs[1] must match result[1]
        if let (Dimension::Static(rhs_n), Dimension::Static(result_n)) =
            (&rhs_shape[1], &result_shape[1])
            && rhs_n != result_n
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::ResultDimMismatch {
                    dim: 1,
                    result_d: *result_n,
                    expected: *rhs_n
                }
            );
        }

        Ok(())
    }
}

impl MatMulOp {
    /// Create a new [MatMulOp], inferring the result type from the input shapes.
    /// The result has shape [M, N] where `lhs` has shape [M, K] and `rhs` has shape [K, N].
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        use crate::memref::type_interfaces::Dimension;
        let result_ty = {
            let lhs_ty = lhs.get_type(ctx);
            let rhs_ty = rhs.get_type(ctx);
            let (elem_ty, m, n) = {
                let lhs_ty_ref = lhs_ty.deref(ctx);
                let rhs_ty_ref = rhs_ty.deref(ctx);
                let lhs_ranked = lhs_ty_ref
                    .downcast_ref::<RankedTensorType>()
                    .expect("MatMulOp lhs must be a RankedTensorType");
                let rhs_ranked = rhs_ty_ref
                    .downcast_ref::<RankedTensorType>()
                    .expect("MatMulOp rhs must be a RankedTensorType");
                let elem_ty = lhs_ranked.element_type();
                let m: Dimension = lhs_ranked.shape()[0].clone();
                let n: Dimension = rhs_ranked.shape()[1].clone();
                (elem_ty, m, n)
            };
            RankedTensorType::get(ctx, elem_ty, vec![m, n])
        };
        Self::new_with_result_type(ctx, lhs, rhs, result_ty.into())
    }

    /// Create a new [MatMulOp] with an explicitly provided result type.
    pub fn new_with_result_type(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        res_ty: Ptr<TypeObj>,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![res_ty],
            vec![lhs, rhs],
            vec![],
            0,
        );
        Self { op }
    }
}
