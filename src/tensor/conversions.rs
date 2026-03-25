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
        inserter::{BlockInsertionPoint, Inserter},
        rewriter::Rewriter,
    },
    linked_list::ContainsLinkedList,
    op::{Op, op_cast, op_impls},
    operation::Operation,
    region::Region,
    result::Result,
    r#type::{TypeObj, TypePtr, Typed, type_cast, type_impls},
    value::Value,
};
use pliron_common_dialects::cf::op_interfaces::YieldingRegion;
use pliron_llvm::ops::FuncOp;

use crate::{
    memref::{
        self, ToMemrefDialect, ToMemrefType, ToMemrefTypeFn, descriptor,
        op_interfaces::ElementWiseBinaryMemrefOpInterface,
        ops::{
            AllocOp, CopyOp as MemrefCopyOp, MatMulOp as MemrefMatMulOp,
            ReshapeOp as MemrefReshapeOp, SubviewOp as MemrefSubviewOp, YieldOp,
        },
        type_interfaces::{Dimension, MultiDimensionalType, ShapedType},
        types::RankedMemrefType,
    },
    tensor::{
        op_interfaces::ElementWiseBinaryTensorOpInterface,
        ops::{
            AddOp, DivOp, ExtractOp, ExtractSliceOp as TensorExtractSliceOp, GenerateOp,
            InsertSliceOp as TensorInsertSliceOp, MatMulOp, MulOp, ReshapeOp as TensorReshapeOp,
            SubOp,
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
    fn can_convert_op(&mut self, ctx: &Context, op: Ptr<Operation>) -> bool {
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

    fn can_convert_type(&mut self, _ctx: &Context, ty: Ptr<TypeObj>) -> bool {
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
