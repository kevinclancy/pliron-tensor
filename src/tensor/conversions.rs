//! Translate tensor to memref

use pliron::{
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{OneOpdInterface, OneRegionInterface, OneResultInterface},
        type_interfaces::FunctionTypeInterface,
    },
    context::{Context, Ptr},
    derive::{op_interface_impl, type_interface_impl},
    irbuild::{
        dialect_conversion::{DialectConversion, DialectConversionRewriter, OperandsInfo},
        inserter::{BlockInsertionPoint, Inserter, OpInsertionPoint},
        listener::Recorder,
        rewriter::{IRRewriter, Rewriter, ScopedRewriter},
    },
    linked_list::ContainsLinkedList,
    op::{Op, op_cast, op_impls},
    operation::Operation,
    region::Region,
    result::Result,
    r#type::{TypeObj, TypePtr, Typed, type_cast, type_impls},
    value::Value,
};
use pliron_common_dialects::{
    cf::{op_interfaces::YieldingRegion, ops::NDForOp},
    index::ops::IndexConstantOp,
};
use pliron_llvm::ops::FuncOp;

use crate::{
    memref::{
        self, ToMemrefDialect, ToMemrefType, ToMemrefTypeFn, descriptor,
        op_interfaces::ElementWiseBinaryMemrefOpInterface,
        ops::{
            AllocOp, CopyOp as MemrefCopyOp, MatMulOp as MemrefMatMulOp,
            ReshapeOp as MemrefReshapeOp, SliceParam, SubviewOp as MemrefSubviewOp, YieldOp,
        },
        type_interfaces::{Dimension, MultiDimensionalType, ShapedType},
        types::RankedMemrefType,
    },
    tensor::{
        op_interfaces::ElementWiseBinaryTensorOpInterface,
        ops::{
            AddOp, BatchMatMulOp, DivOp, ExtractOp, ExtractSliceOp as TensorExtractSliceOp,
            GenerateOp, InsertSliceOp as TensorInsertSliceOp, MatMulOp, MulOp,
            ReshapeOp as TensorReshapeOp, SubOp,
        },
        types::RankedTensorType,
    },
};

#[type_interface_impl]
impl ToMemrefType for RankedTensorType {
    fn converter(&self) -> ToMemrefTypeFn {
        |self_ty, ctx| {
            let (element_ty, shape) = {
                let ranked_tensor_ty = self_ty.deref(ctx);
                let ranked_tensor_ty = ranked_tensor_ty
                    .downcast_ref::<RankedTensorType>()
                    .expect("Expected a RankedTensorType");
                (
                    ranked_tensor_ty.element_type(),
                    ranked_tensor_ty.shape().clone(),
                )
            };
            let memref_ty = RankedMemrefType::get(ctx, element_ty, shape);
            Ok(memref_ty.into())
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum GenerateOpConversionErr {
    #[error("Unsupported induction variable type for GenerateOp conversion")]
    UnsupportedIVType,
}

#[op_interface_impl]
impl ToMemrefDialect for GenerateOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        let result_ty_ptr = self.get_result(ctx).get_type(ctx);
        let converter = {
            let result_ty_ref = result_ty_ptr.deref(ctx);
            let result_ty = result_ty_ref
                .downcast_ref::<RankedTensorType>()
                .expect("GenerateOp must have a ranked tensor result");
            result_ty.converter()
        };
        let result_ty = converter(result_ty_ptr, ctx)?;
        let result_ty = TypePtr::<RankedMemrefType>::from_ptr(result_ty, ctx)
            .expect("Expected the converted type to be a RankedMemrefType");

        let region = self.get_region(ctx);

        let alloc = AllocOp::new(ctx, result_ty, self.dynamic_dimensions(ctx).clone());
        rewriter.append_op(ctx, alloc);

        let yield_op = self.get_yield(ctx);

        struct State<'a> {
            yield_op: YieldOp,
            rewriter: &'a mut DialectConversionRewriter,
            inline_region: Ptr<Region>,
        }
        let generate_op = memref::ops::GenerateOp::new(
            ctx,
            alloc.get_result(ctx),
            |ctx, state, inserter, indices: Vec<Value>| {
                let previos_entry = state
                    .inline_region
                    .deref(ctx)
                    .get_head()
                    .expect("Region must have at least one block");
                state.rewriter.inline_region(
                    ctx,
                    state.inline_region,
                    BlockInsertionPoint::AfterBlock(
                        inserter
                            .get_insertion_block(ctx)
                            .expect("Inserter must be set to entry block"),
                    ),
                );
                let branch = pliron_llvm::ops::BrOp::new(ctx, previos_entry, indices);
                inserter.append_op(ctx, branch);
                let yield_value = state.yield_op.get_operand(ctx);
                // Remove the previous yield as the memref GenerateOp will add a new one.
                state
                    .rewriter
                    .erase_operation(ctx, state.yield_op.get_operation());
                yield_value
            },
            State {
                yield_op,
                rewriter,
                inline_region: region,
            },
        );
        rewriter.append_op(ctx, generate_op);
        rewriter.replace_operation(ctx, self.get_operation(), alloc.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToMemrefDialect for ExtractOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        let operand = self.get_tensor_operand(ctx);
        let indices = self.get_index_operands(ctx);
        let result_ty = self.get_result(ctx).get_type(ctx);

        // Create a LoadOp to extract the value from the memref.
        let load_op = memref::ops::LoadOp::new(ctx, result_ty, operand, indices.clone());
        rewriter.append_op(ctx, load_op);
        rewriter.replace_operation(ctx, self.get_operation(), load_op.get_operation());
        Ok(())
    }
}

trait ElementWiseBinaryTensorOpToMemref: ElementWiseBinaryTensorOpInterface {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        let lhs = self.get_operation().deref(ctx).get_operand(0);
        let rhs = self.get_operation().deref(ctx).get_operand(1);

        let result_ty_ptr = self.get_result(ctx).get_type(ctx);
        let converter = {
            let result_ty_ref = result_ty_ptr.deref(ctx);
            let result_ty = result_ty_ref
                .downcast_ref::<RankedTensorType>()
                .expect("AddOp must have a ranked tensor result");
            result_ty.converter()
        };
        let result_ty = converter(result_ty_ptr, ctx)?;
        let result_ty = TypePtr::<RankedMemrefType>::from_ptr(result_ty, ctx)
            .expect("Expected the converted type to be a RankedMemrefType");
        let elem_ty = result_ty.deref(ctx).element_type();
        // Based on the operand shapes, it is possible that the result shape can be inferred
        // to have more static dimensions than what we know with `result_ty` above.
        let compatible_shape = self.compatible_shape(ctx);
        let dynamic_dim_operands = compatible_shape
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| {
                if let Dimension::Dynamic = dim {
                    // Get the dynamic operands from the memref descriptor of the first operand.
                    Some(descriptor::unpack_size(ctx, rewriter, lhs, i))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let result_ty = RankedMemrefType::get(ctx, elem_ty, compatible_shape);

        let alloc = AllocOp::new(ctx, result_ty, dynamic_dim_operands);
        rewriter.append_op(ctx, alloc);
        let add = self.build_memref_op(ctx, alloc.get_result(ctx), lhs, rhs);
        rewriter.append_operation(ctx, add);
        rewriter.replace_operation(ctx, self.get_operation(), alloc.get_operation());
        Ok(())
    }

    fn build_memref_op(
        &self,
        ctx: &mut Context,
        res: Value,
        lhs: Value,
        rhs: Value,
    ) -> Ptr<Operation>;
}

impl ElementWiseBinaryTensorOpToMemref for AddOp {
    fn build_memref_op(
        &self,
        ctx: &mut Context,
        res: Value,
        lhs: Value,
        rhs: Value,
    ) -> Ptr<Operation> {
        memref::ops::AddOp::new(ctx, res, lhs, rhs).get_operation()
    }
}

impl ElementWiseBinaryTensorOpToMemref for SubOp {
    fn build_memref_op(
        &self,
        ctx: &mut Context,
        res: Value,
        lhs: Value,
        rhs: Value,
    ) -> Ptr<Operation> {
        memref::ops::SubOp::new(ctx, res, lhs, rhs).get_operation()
    }
}

impl ElementWiseBinaryTensorOpToMemref for MulOp {
    fn build_memref_op(
        &self,
        ctx: &mut Context,
        res: Value,
        lhs: Value,
        rhs: Value,
    ) -> Ptr<Operation> {
        memref::ops::MulOp::new(ctx, res, lhs, rhs).get_operation()
    }
}

impl ElementWiseBinaryTensorOpToMemref for DivOp {
    fn build_memref_op(
        &self,
        ctx: &mut Context,
        res: Value,
        lhs: Value,
        rhs: Value,
    ) -> Ptr<Operation> {
        memref::ops::DivOp::new(ctx, res, lhs, rhs).get_operation()
    }
}

#[op_interface_impl]
impl ToMemrefDialect for AddOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        <Self as ElementWiseBinaryTensorOpToMemref>::rewrite(self, ctx, rewriter)
    }
}

#[op_interface_impl]
impl ToMemrefDialect for SubOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        <Self as ElementWiseBinaryTensorOpToMemref>::rewrite(self, ctx, rewriter)
    }
}

#[op_interface_impl]
impl ToMemrefDialect for MulOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        <Self as ElementWiseBinaryTensorOpToMemref>::rewrite(self, ctx, rewriter)
    }
}

#[op_interface_impl]
impl ToMemrefDialect for DivOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        <Self as ElementWiseBinaryTensorOpToMemref>::rewrite(self, ctx, rewriter)
    }
}

#[op_interface_impl]
impl ToMemrefDialect for pliron_llvm::ops::LoadOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        let loaded_ty = self.get_result(ctx).get_type(ctx);
        let to_memref_ty =
            type_cast::<dyn ToMemrefType>(&**loaded_ty.deref(ctx)).map(|t| t.converter());
        let memref_ty = if let Some(to_memref_ty) = to_memref_ty {
            (to_memref_ty)(loaded_ty, ctx)?
        } else {
            loaded_ty
        };
        rewriter.set_value_type(ctx, self.get_result(ctx), memref_ty);
        Ok(())
    }
}

// Lowering for tensor::MatMulOp -> AllocOp + memref::MatMulOp
#[op_interface_impl]
impl ToMemrefDialect for MatMulOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        let lhs = self.get_operation().deref(ctx).get_operand(0);
        let rhs = self.get_operation().deref(ctx).get_operand(1);

        let result_ty_ptr = self.get_result(ctx).get_type(ctx);
        let converter = {
            let result_ty_ref = result_ty_ptr.deref(ctx);
            let result_ty = result_ty_ref
                .downcast_ref::<RankedTensorType>()
                .expect("MatMulOp must have a ranked tensor result");
            result_ty.converter()
        };
        let result_ty = converter(result_ty_ptr, ctx)?;
        let result_ty = TypePtr::<RankedMemrefType>::from_ptr(result_ty, ctx)
            .expect("Expected the converted type to be a RankedMemrefType");
        let elem_ty = result_ty.deref(ctx).element_type();

        // Build the dynamic dimension operands for the result allocation.
        // Result shape: [M, N] where M = lhs dim 0, N = rhs dim 1.
        let result_shape = result_ty.deref(ctx).shape().clone();
        let dynamic_dim_operands = result_shape
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| {
                if let Dimension::Dynamic = dim {
                    let val = if i == 0 {
                        descriptor::unpack_size(ctx, rewriter, lhs, 0)
                    } else {
                        descriptor::unpack_size(ctx, rewriter, rhs, 1)
                    };
                    Some(val)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let result_memref_ty = RankedMemrefType::get(ctx, elem_ty, result_shape);
        let alloc = AllocOp::new(ctx, result_memref_ty, dynamic_dim_operands);
        rewriter.append_op(ctx, alloc);

        let matmul = MemrefMatMulOp::new(ctx, alloc.get_result(ctx), lhs, rhs);
        rewriter.append_operation(ctx, matmul.get_operation());

        rewriter.replace_operation(ctx, self.get_operation(), alloc.get_operation());
        Ok(())
    }
}

// Lowering for tensor::BatchMatMulOp.
// Creates a destination memref and then iterates over batch dimensions using NDForOp.
// For each batch index, it creates subviews of lhs/rhs/result and performs 2D matmul.
#[op_interface_impl]
impl ToMemrefDialect for BatchMatMulOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        use crate::memref::type_interfaces::Dimension;

        let lhs = self.get_operation().deref(ctx).get_operand(0);
        let rhs = self.get_operation().deref(ctx).get_operand(1);

        let result_ty_ptr = self.get_result(ctx).get_type(ctx);
        let converter = {
            let result_ty_ref = result_ty_ptr.deref(ctx);
            let result_ty = result_ty_ref
                .downcast_ref::<RankedTensorType>()
                .expect("BatchMatMulOp must have a ranked tensor result");
            result_ty.converter()
        };
        let result_ty = converter(result_ty_ptr, ctx)?;
        let result_ty = TypePtr::<RankedMemrefType>::from_ptr(result_ty, ctx)
            .expect("Expected the converted type to be a RankedMemrefType");

        let lhs_memref_ty = TypePtr::<RankedMemrefType>::from_ptr(lhs.get_type(ctx), ctx)
            .expect("BatchMatMulOp lhs must be a ranked memref after conversion");
        let rhs_memref_ty = TypePtr::<RankedMemrefType>::from_ptr(rhs.get_type(ctx), ctx)
            .expect("BatchMatMulOp rhs must be a ranked memref after conversion");

        let rank = lhs_memref_ty.deref(ctx).rank();
        let batch_rank = rank - 2;

        let lhs_shape = lhs_memref_ty.deref(ctx).shape().clone();
        let rhs_shape = rhs_memref_ty.deref(ctx).shape().clone();
        let result_shape = result_ty.deref(ctx).shape().clone();
        let elem_ty = result_ty.deref(ctx).element_type();

        let lhs_sizes = lhs_shape
            .iter()
            .enumerate()
            .map(|(i, dim)| match dim {
                Dimension::Dynamic => descriptor::unpack_size(ctx, rewriter, lhs, i),
                Dimension::Static(v) => {
                    let c = IndexConstantOp::new(ctx, *v);
                    rewriter.append_op(ctx, c);
                    c.get_result(ctx)
                }
            })
            .collect::<Vec<_>>();
        let rhs_sizes = rhs_shape
            .iter()
            .enumerate()
            .map(|(i, dim)| match dim {
                Dimension::Dynamic => descriptor::unpack_size(ctx, rewriter, rhs, i),
                Dimension::Static(v) => {
                    let c = IndexConstantOp::new(ctx, *v);
                    rewriter.append_op(ctx, c);
                    c.get_result(ctx)
                }
            })
            .collect::<Vec<_>>();

        let result_dynamic_dim_operands = result_shape
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| {
                if let Dimension::Dynamic = dim {
                    if i < batch_rank {
                        Some(lhs_sizes[i])
                    } else if i == batch_rank {
                        Some(lhs_sizes[rank - 2])
                    } else {
                        Some(rhs_sizes[rank - 1])
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let alloc = AllocOp::new(ctx, result_ty, result_dynamic_dim_operands);
        rewriter.append_op(ctx, alloc);

        if batch_rank == 0 {
            let matmul = MemrefMatMulOp::new(ctx, alloc.get_result(ctx), lhs, rhs);
            rewriter.append_operation(ctx, matmul.get_operation());
            rewriter.replace_operation(ctx, self.get_operation(), alloc.get_operation());
            return Ok(());
        }

        let const_index_0 = IndexConstantOp::new(ctx, 0);
        let const_index_1 = IndexConstantOp::new(ctx, 1);
        rewriter.append_op(ctx, const_index_0);
        rewriter.append_op(ctx, const_index_1);

        let lb0 = const_index_0.get_result(ctx);
        let step1 = const_index_1.get_result(ctx);

        let batch_ubs = (0..batch_rank).map(|i| lhs_sizes[i]).collect::<Vec<_>>();

        let ndfor = {
            let scoped_rewriter = ScopedRewriter::new(rewriter, OpInsertionPoint::Unset);

            struct State<'a> {
                rewriter: ScopedRewriter<'a, Recorder, IRRewriter<Recorder>>,
                lhs: Value,
                rhs: Value,
                dst: Value,
                lhs_shape: Vec<Dimension>,
                rhs_shape: Vec<Dimension>,
                lhs_sizes: Vec<Value>,
                rhs_sizes: Vec<Value>,
                rank: usize,
                batch_rank: usize,
                elem_ty: Ptr<TypeObj>,
            }

            let mut state = State {
                rewriter: scoped_rewriter,
                lhs,
                rhs,
                dst: alloc.get_result(ctx),
                lhs_shape,
                rhs_shape,
                lhs_sizes,
                rhs_sizes,
                rank,
                batch_rank,
                elem_ty,
            };

            NDForOp::new(
                ctx,
                vec![lb0; batch_rank],
                batch_ubs,
                vec![step1; batch_rank],
                |ctx, state, inserter, indices| {
                    let rewriter = &mut state.rewriter;
                    rewriter.set_insertion_point(inserter.get_insertion_point());

                    let dim_to_size = |dim: &Dimension, size_val: Value| match dim {
                        Dimension::Static(v) => SliceParam::Static(*v),
                        Dimension::Dynamic => SliceParam::Dynamic(size_val),
                    };

                    let one_sizes = vec![SliceParam::Static(1); state.batch_rank];
                    let zero_offsets = vec![SliceParam::Static(0); 2];
                    let unit_steps = vec![SliceParam::Static(1); state.rank];

                    let lhs_m_dim = dim_to_size(
                        &state.lhs_shape[state.rank - 2],
                        state.lhs_sizes[state.rank - 2],
                    );
                    let lhs_k_dim = dim_to_size(
                        &state.lhs_shape[state.rank - 1],
                        state.lhs_sizes[state.rank - 1],
                    );
                    let rhs_k_dim = dim_to_size(
                        &state.rhs_shape[state.rank - 2],
                        state.rhs_sizes[state.rank - 2],
                    );
                    let rhs_n_dim = dim_to_size(
                        &state.rhs_shape[state.rank - 1],
                        state.rhs_sizes[state.rank - 1],
                    );

                    let mut lhs_offsets = indices
                        .iter()
                        .copied()
                        .map(SliceParam::Dynamic)
                        .collect::<Vec<_>>();
                    lhs_offsets.extend(zero_offsets.clone());
                    let mut lhs_sizes = one_sizes.clone();
                    lhs_sizes.push(lhs_m_dim.clone());
                    lhs_sizes.push(lhs_k_dim.clone());

                    let lhs_subview = MemrefSubviewOp::new(
                        ctx,
                        state.lhs,
                        lhs_offsets,
                        lhs_sizes,
                        unit_steps.clone(),
                    );
                    rewriter.append_op(ctx, lhs_subview);

                    let mut rhs_offsets = indices
                        .iter()
                        .copied()
                        .map(SliceParam::Dynamic)
                        .collect::<Vec<_>>();
                    rhs_offsets.extend(zero_offsets.clone());
                    let mut rhs_sizes = one_sizes.clone();
                    rhs_sizes.push(rhs_k_dim.clone());
                    rhs_sizes.push(rhs_n_dim.clone());

                    let rhs_subview = MemrefSubviewOp::new(
                        ctx,
                        state.rhs,
                        rhs_offsets,
                        rhs_sizes,
                        unit_steps.clone(),
                    );
                    rewriter.append_op(ctx, rhs_subview);

                    let mut dst_offsets = indices
                        .iter()
                        .copied()
                        .map(SliceParam::Dynamic)
                        .collect::<Vec<_>>();
                    dst_offsets.extend(zero_offsets);
                    let mut dst_sizes = one_sizes;
                    dst_sizes.push(lhs_m_dim.clone());
                    dst_sizes.push(rhs_n_dim.clone());

                    let dst_subview =
                        MemrefSubviewOp::new(ctx, state.dst, dst_offsets, dst_sizes, unit_steps);
                    rewriter.append_op(ctx, dst_subview);

                    let m_dim = match &lhs_m_dim {
                        SliceParam::Static(v) => Dimension::Static(*v),
                        SliceParam::Dynamic(_) => Dimension::Dynamic,
                    };
                    let k_dim = match &lhs_k_dim {
                        SliceParam::Static(v) => Dimension::Static(*v),
                        SliceParam::Dynamic(_) => Dimension::Dynamic,
                    };
                    let n_dim = match &rhs_n_dim {
                        SliceParam::Static(v) => Dimension::Static(*v),
                        SliceParam::Dynamic(_) => Dimension::Dynamic,
                    };

                    let lhs_2d_ty = RankedMemrefType::get(
                        ctx,
                        state.elem_ty,
                        vec![m_dim.clone(), k_dim.clone()],
                    );
                    let rhs_2d_ty = RankedMemrefType::get(
                        ctx,
                        state.elem_ty,
                        vec![k_dim.clone(), n_dim.clone()],
                    );
                    let dst_2d_ty = RankedMemrefType::get(
                        ctx,
                        state.elem_ty,
                        vec![m_dim.clone(), n_dim.clone()],
                    );

                    let dyn_value = |p: &SliceParam| match p {
                        SliceParam::Dynamic(v) => Some(*v),
                        SliceParam::Static(_) => None,
                    };

                    let mut lhs_2d_dyn = Vec::new();
                    if let Some(v) = dyn_value(&lhs_m_dim) {
                        lhs_2d_dyn.push(v);
                    }
                    if let Some(v) = dyn_value(&lhs_k_dim) {
                        lhs_2d_dyn.push(v);
                    }

                    let mut rhs_2d_dyn = Vec::new();
                    if let Some(v) = dyn_value(&rhs_k_dim) {
                        rhs_2d_dyn.push(v);
                    }
                    if let Some(v) = dyn_value(&rhs_n_dim) {
                        rhs_2d_dyn.push(v);
                    }

                    let mut dst_2d_dyn = Vec::new();
                    if let Some(v) = dyn_value(&lhs_m_dim) {
                        dst_2d_dyn.push(v);
                    }
                    if let Some(v) = dyn_value(&rhs_n_dim) {
                        dst_2d_dyn.push(v);
                    }

                    let lhs_2d = MemrefReshapeOp::new(
                        ctx,
                        lhs_subview.get_result(ctx),
                        lhs_2d_dyn,
                        lhs_2d_ty,
                    );
                    rewriter.append_op(ctx, lhs_2d);

                    let rhs_2d = MemrefReshapeOp::new(
                        ctx,
                        rhs_subview.get_result(ctx),
                        rhs_2d_dyn,
                        rhs_2d_ty,
                    );
                    rewriter.append_op(ctx, rhs_2d);

                    let dst_2d = MemrefReshapeOp::new(
                        ctx,
                        dst_subview.get_result(ctx),
                        dst_2d_dyn,
                        dst_2d_ty,
                    );
                    rewriter.append_op(ctx, dst_2d);

                    let matmul = MemrefMatMulOp::new(
                        ctx,
                        dst_2d.get_result(ctx),
                        lhs_2d.get_result(ctx),
                        rhs_2d.get_result(ctx),
                    );
                    rewriter.append_operation(ctx, matmul.get_operation());
                },
                &mut state,
            )
        };

        rewriter.append_op(ctx, ndfor);
        rewriter.replace_operation(ctx, self.get_operation(), alloc.get_operation());
        Ok(())
    }
}

fn lower_func_op_to_llvm(func_op: &FuncOp, ctx: &mut Context) -> Result<()> {
    // update the function type to convert any tensor types in the signature to memref types.
    let func_ty = func_op.get_type(ctx);
    let res_ty = func_ty.deref(ctx).result_type();
    let res_ty_converter = type_cast::<dyn ToMemrefType>(&**res_ty.deref(ctx))
        .map(|to_memref_ty| to_memref_ty.converter());
    let res_ty = if let Some(res_ty_converter) = res_ty_converter {
        (res_ty_converter)(res_ty, ctx)?
    } else {
        res_ty
    };
    let arg_tys = func_ty.deref(ctx).arg_types();
    let arg_tys = arg_tys
        .iter()
        .map(|arg_ty| {
            let arg_ty_converter = type_cast::<dyn ToMemrefType>(&**arg_ty.deref(ctx))
                .map(|to_memref_ty| to_memref_ty.converter());
            if let Some(arg_ty_converter) = arg_ty_converter {
                (arg_ty_converter)(*arg_ty, ctx)
            } else {
                Ok(*arg_ty)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let new_func_ty = pliron_llvm::types::FuncType::get(ctx, res_ty, arg_tys, false);
    func_op.set_attr_llvm_func_type(ctx, TypeAttr::new(new_func_ty.into()));

    // Update all arguments in the entry block to use the new memref types.
    let entry_block = func_op
        .get_entry_block(ctx)
        .expect("FuncOp must have an entry block");

    let args = entry_block.deref(ctx).arguments().collect::<Vec<_>>();
    for arg in args {
        let arg_ty = arg.get_type(ctx);
        let arg_ty_converter = type_cast::<dyn ToMemrefType>(&**arg_ty.deref(ctx))
            .map(|to_memref_ty| to_memref_ty.converter());
        let arg_ty = if let Some(arg_ty_converter) = arg_ty_converter {
            (arg_ty_converter)(arg_ty, ctx)?
        } else {
            arg_ty
        };
        arg.set_type(ctx, arg_ty);
    }

    Ok(())
}

// Lowering for tensor::ExtractSliceOp -> memref.alloc + memref.subview + memref.copy
#[op_interface_impl]
impl ToMemrefDialect for TensorExtractSliceOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        let result_ty_ptr = self.get_result(ctx).get_type(ctx);
        let converter = {
            let result_ty_ref = result_ty_ptr.deref(ctx);
            let result_ty = result_ty_ref
                .downcast_ref::<RankedTensorType>()
                .expect("ExtractSliceOp must have a ranked tensor result");
            result_ty.converter()
        };
        let result_ty = converter(result_ty_ptr, ctx)?;
        let result_ty = TypePtr::<RankedMemrefType>::from_ptr(result_ty, ctx)
            .expect("Expected the converted type to be a RankedMemrefType");

        let source = self.source(ctx);
        let offsets = self.slice_offsets(ctx);
        let sizes = self.slice_sizes(ctx);
        let steps = self.slice_steps(ctx);

        let dynamic_dim_operands = sizes
            .iter()
            .filter_map(|size| match size {
                memref::ops::SliceParam::Dynamic(v) => Some(*v),
                memref::ops::SliceParam::Static(_) => None,
            })
            .collect::<Vec<_>>();

        let destination = AllocOp::new(ctx, result_ty, dynamic_dim_operands);
        rewriter.append_op(ctx, destination);

        let subview =
            MemrefSubviewOp::new_with_result_type(ctx, source, offsets, sizes, steps, result_ty);
        rewriter.append_op(ctx, subview);

        let copy = MemrefCopyOp::new(ctx, destination.get_result(ctx), subview.get_result(ctx));
        rewriter.append_op(ctx, copy);
        rewriter.replace_operation(ctx, self.get_operation(), destination.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToMemrefDialect for TensorInsertSliceOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        let result_ty_ptr = self.get_result(ctx).get_type(ctx);
        let converter = {
            let result_ty_ref = result_ty_ptr.deref(ctx);
            let result_ty = result_ty_ref
                .downcast_ref::<RankedTensorType>()
                .expect("InsertSliceOp must have a ranked tensor result");
            result_ty.converter()
        };
        let result_ty = converter(result_ty_ptr, ctx)?;
        let result_ty = TypePtr::<RankedMemrefType>::from_ptr(result_ty, ctx)
            .expect("Expected the converted type to be a RankedMemrefType");

        let source = self.source(ctx);
        let destination = self.destination(ctx);

        let result_shape = result_ty.deref(ctx).shape().clone();
        let dynamic_dim_operands = result_shape
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| {
                if let Dimension::Dynamic = dim {
                    Some(descriptor::unpack_size(ctx, rewriter, destination, i))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let alloc = AllocOp::new(ctx, result_ty, dynamic_dim_operands);
        rewriter.append_op(ctx, alloc);

        // Preserve destination values in the new result memref, then overwrite the sliced window.
        let copy_destination = MemrefCopyOp::new(ctx, alloc.get_result(ctx), destination);
        rewriter.append_op(ctx, copy_destination);

        let view = MemrefSubviewOp::new(
            ctx,
            alloc.get_result(ctx),
            self.slice_offsets(ctx),
            self.slice_sizes(ctx),
            self.slice_steps(ctx),
        );
        rewriter.append_op(ctx, view);

        let copy_source = MemrefCopyOp::new(ctx, view.get_result(ctx), source);
        rewriter.append_op(ctx, copy_source);
        rewriter.replace_operation(ctx, self.get_operation(), alloc.get_operation());
        Ok(())
    }
}

// Lowering for tensor::ReshapeOp -> memref.alloc + memref.copy + memref.reshape
#[op_interface_impl]
impl ToMemrefDialect for TensorReshapeOp {
    fn rewrite(&self, ctx: &mut Context, rewriter: &mut DialectConversionRewriter) -> Result<()> {
        let source = self.get_source(ctx);
        let source_memref_ty = TypePtr::<RankedMemrefType>::from_ptr(source.get_type(ctx), ctx)
            .expect("ReshapeOp source must be a ranked memref after tensor-to-memref conversion");
        let result_ty_ptr = self.get_result(ctx).get_type(ctx);
        let converter = {
            let result_ty_ref = result_ty_ptr.deref(ctx);
            let result_ty = result_ty_ref
                .downcast_ref::<RankedTensorType>()
                .expect("ReshapeOp must have a ranked tensor result");
            result_ty.converter()
        };
        let result_ty = converter(result_ty_ptr, ctx)?;
        let result_ty = TypePtr::<RankedMemrefType>::from_ptr(result_ty, ctx)
            .expect("Expected the converted type to be a RankedMemrefType");

        // Create a contiguous copy of the source first, then reshape that copy by
        // constructing a descriptor with the target shape.
        let source_shape = source_memref_ty.deref(ctx).shape().clone();
        let source_dyn_dims = source_shape
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| {
                if let Dimension::Dynamic = dim {
                    Some(descriptor::unpack_size(ctx, rewriter, source, i))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let source_copy_alloc = AllocOp::new(ctx, source_memref_ty, source_dyn_dims);
        rewriter.append_op(ctx, source_copy_alloc);

        let copy_source = MemrefCopyOp::new(ctx, source_copy_alloc.get_result(ctx), source);
        rewriter.append_op(ctx, copy_source);

        let memref_reshape = MemrefReshapeOp::new(
            ctx,
            source_copy_alloc.get_result(ctx),
            self.get_dynamic_dimensions(ctx),
            result_ty,
        );
        rewriter.append_op(ctx, memref_reshape);

        rewriter.replace_operation(ctx, self.get_operation(), memref_reshape.get_operation());
        Ok(())
    }
}

/// Implement [DialectConversion] for tensor to memref conversion.
pub struct TensorToMemref;

impl DialectConversion for TensorToMemref {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn ToMemrefDialect>(&*Operation::get_op_dyn(op, ctx))
            || Operation::get_op::<FuncOp>(op, ctx).is_some()
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        if let Some(func_op) = Operation::get_op::<FuncOp>(op, ctx) {
            return lower_func_op_to_llvm(&func_op, ctx);
        }
        let op_dyn = Operation::get_op_dyn(op, ctx);
        let to_memref_op = op_cast::<dyn ToMemrefDialect>(&*op_dyn)
            .expect("Matched Op must implement ToMemrefDialect");
        to_memref_op.rewrite(ctx, rewriter)
    }

    fn can_convert_type(&self, _ctx: &Context, ty: Ptr<TypeObj>) -> bool {
        type_impls::<dyn ToMemrefType>(&**ty.deref(_ctx))
    }

    fn convert_type(&mut self, _ctx: &mut Context, ty: Ptr<TypeObj>) -> Result<Ptr<TypeObj>> {
        let to_memref_ty = type_cast::<dyn ToMemrefType>(&**ty.deref(_ctx)).map(|t| t.converter());
        if let Some(to_memref_ty) = to_memref_ty {
            to_memref_ty(ty, _ctx)
        } else {
            Ok(ty)
        }
    }
}
