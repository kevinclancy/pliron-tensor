//! Tensor semantics -> memref semantics

use rustc_hash::FxHashSet;

use pliron::{
    analyses::liveness::{Liveness, LivenessTq},
    builtin::op_interfaces::{AtLeastNOpdsInterface, OneResultInterface, ResultNOfType},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::op_interface,
    graph::walkers::{self, IRNode},
    irbuild::{
        dialect_conversion::{
            DialectConversion, DialectConversionRewriter, OperandsInfo, apply_dialect_conversion,
        },
        inserter::{Inserter, OpInsertionPoint},
    },
    op::{Op, op_cast, op_impls},
    operation::Operation,
    result::Result,
    r#type::{TypeObj, TypePtr, Typed, type_cast, type_impls},
    value::{Use, Value},
    verify_err_noloc,
};
use pliron_common_dialects::index::{ops::IndexConstantOp, types::IndexType};
use pliron_llvm::{ToLLVMDialect, ops::FuncOp};
use thiserror::Error;

use crate::{
    memref::{
        ToMemrefType,
        ops::{CopyOp, DimOp},
        type_interfaces::{Dimension, ShapedType},
        types::RankedMemrefType,
    },
    tensor::conversions::lower_func_op_to_llvm,
};

/// Is an alias May or Must
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasKind {
    May,
    Must,
}

/// Buffer relation
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferRelation {
    /// Relationship b/w the operand and result buffers is unknown.
    #[default]
    Unknown,
    /// Operand buffer may contain the result buffer.
    Contains,
    /// Operand buffer may be contained in the result buffer.
    Contained,
    /// Operand buffer and result buffer are equivalent.
    Equivalent,
}

/// Describes aliasing b/w operands and results.
pub struct Alias {
    /// The operand that may alias with [Self::result].
    pub operand: Use<Value>,
    /// The result that may alias with [Self::operand].
    pub result: Value,
    /// Alias kind (May or Must)
    pub kind: AliasKind,
    /// Buffer relation b/w the operand and result buffers.
    pub relation: BufferRelation,
}

impl Alias {
    /// Get all results aliasing with `opd`.
    pub fn get_aliases_for_operand(aliases: &[Alias], opd: Use<Value>) -> Vec<&Alias> {
        aliases
            .iter()
            .filter(|alias| alias.operand == opd)
            .collect::<Vec<_>>()
    }

    /// Get all operands aliasing with `res`.
    pub fn get_aliases_for_result(aliases: &[Alias], res: Value) -> Vec<&Alias> {
        aliases
            .iter()
            .filter(|alias| alias.result == res)
            .collect::<Vec<_>>()
    }

    /// Does the operand alias with multiple results?
    pub fn operand_aliases_with_multiple_results(aliases: &[Alias], opd: Use<Value>) -> bool {
        Self::get_aliases_for_operand(aliases, opd).len() > 1
    }

    /// Does this operand alias with other operands?
    pub fn operand_aliases_with_other_operands(aliases: &[Alias], opd: Use<Value>) -> bool {
        let aliasing_results: Vec<_> = Self::get_aliases_for_operand(aliases, opd)
            .iter()
            .map(|alias| alias.result)
            .collect();
        aliases
            .iter()
            .any(|alias| aliasing_results.contains(&alias.result) && alias.operand != opd)
    }
}

#[derive(Debug, Error)]

pub enum AliasErr {
    #[error("Invalid alias: the operand and result do not belong to the same op")]
    InvalidAlias,
    #[error(
        "Incorrect number of dynamic dimension operands: Must be equal to number of dynamic dimensions in the operand type"
    )]
    IncorrectNumDynamicDims,
    #[error("Invalid dynamic dimension operand type: Must be of type Index")]
    InvalidDynamicDimOpdType,
    #[error("Operand type is not a shaped type")]
    InvalidOperandType,
}

impl Verify for Alias {
    fn verify(&self, _ctx: &Context) -> Result<()> {
        let Value::OpResult { op, .. } = self.result else {
            return verify_err_noloc!(AliasErr::InvalidAlias);
        };
        if self.operand.user_op != op {
            return verify_err_noloc!(AliasErr::InvalidAlias);
        }
        Ok(())
    }
}

/// [Op]s implementing this can participate in bufferization.
#[op_interface]
pub trait BufferizableOpInterface {
    /// Return true if this operand bufferizes to a memory read of that operand.
    /// It will only be called on operands that have a tensor type.
    ///
    /// It is always safe to return `true`, but that may introduce unnecessary
    /// allocations and / or copies.
    fn operand_bufferizes_to_memory_read(&self, ctx: &Context, opd: Use<Value>) -> bool;

    /// Return true if this operand bufferizes to a memory write of that operand.
    /// It will only be called on operands that have a tensor type.
    ///
    /// It is always safe to return `true`, but that may introduce unnecessary
    /// allocations and / or copies.
    fn operand_bufferizes_to_memory_write(&self, ctx: &Context, opd: Use<Value>) -> bool;

    /// Get post-bufferization aliasing info between this op's operands and results.
    /// If after bufferization, the buffer of an operand may alias with the buffer of a result,
    /// then, the returned vector should contain an [Alias] with the appropriate information.
    fn get_operand_result_aliases(&self, ctx: &Context) -> Vec<Alias>;

    /// Get the dynamic dimensions for the given operand.
    /// On `None`, `memref.dim` will be used (less efficient).
    /// It will only be called on aliasing operands that have a tensor type.
    fn get_dynamic_dimensions(&self, ctx: &Context, opd: Use<Value>) -> Option<Vec<Value>>;

    /// Rewrite to use memref semantics.
    ///
    /// Operands will have already been bufferized (i.e., converted to memrefs).
    /// Non-aliasing results will need to be be bufferized (by allocating a new buffer).
    /// Aliasing results can assume that it's safe to reuse operand buffers.
    ///
    /// `operands_info` semantics are as in [DialectConversion::rewrite], and can be used
    /// to get pre-conversion operand types.
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()>;

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let op = op
            .downcast_ref::<Self>()
            .expect("Failed to downcast op to Self");
        let aliases = op.get_operand_result_aliases(ctx);

        for alias in aliases {
            alias.verify(ctx)?;
            let opd_ty = alias.operand.get_type(ctx);
            let opd_ty = opd_ty.deref(ctx);
            let Some(shaped_ty) = type_cast::<dyn ShapedType>(&**opd_ty) else {
                return verify_err_noloc!(AliasErr::InvalidOperandType);
            };
            let dynamic_dims_opt = op.get_dynamic_dimensions(ctx, alias.operand);
            let num_dynamic_dims = shaped_ty.num_dynamic_dimensions();
            if let Some(dynamic_dims) = dynamic_dims_opt {
                if dynamic_dims.len() != num_dynamic_dims {
                    return verify_err_noloc!(AliasErr::IncorrectNumDynamicDims);
                }
                if !dynamic_dims
                    .iter()
                    .all(|dim| dim.get_type(ctx).deref(ctx).is::<IndexType>())
                {
                    return verify_err_noloc!(AliasErr::InvalidDynamicDimOpdType);
                }
            }
        }
        Ok(())
    }
}

/// An [Op] to Allocate a buffer (memref) for the given tensor type and dynamic sizes (if any).
#[op_interface]
pub trait MemrefAllocOpInterface:
    OneResultInterface + ResultNOfType<0, RankedMemrefType> + ToLLVMDialect
{
    /// Create a new [Self] to allocate a buffer for `tensor_ty` with given `dynamic_sizes`.
    fn try_new(
        ctx: &mut Context,
        memref_ty: TypePtr<RankedMemrefType>,
        dynamic_sizes: Vec<Value>,
    ) -> Result<Self>
    where
        Self: Sized;

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// An [Op] to Deallocate a buffer (memref).
#[op_interface]
pub trait MemrefDeallocOpInterface: AtLeastNOpdsInterface<1> + ToLLVMDialect {
    /// Create a new [Self] to deallocate the buffer in `memref`.
    fn try_new(ctx: &mut Context, memref: Value) -> Result<Self>
    where
        Self: Sized;

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// A helper struct that implements [DialectConversion]
/// to bufferize from tensor semantics to memref semantics.
struct Bufferizer<A: MemrefAllocOpInterface, D: MemrefDeallocOpInterface> {
    alloc_op: std::marker::PhantomData<A>,
    /// TODO: Use this to insert deallocs for dead buffers.
    dealloc_op: std::marker::PhantomData<D>,
    /// Set of operands that can be bufferized in-place (i.e., without copy).
    /// Whether a result (that alises to this operand) needs a new allocation
    /// or not is left to the op's rewrite method.
    in_place_bufferizable_operands: FxHashSet<Use<Value>>,
}

impl<A: MemrefAllocOpInterface, D: MemrefDeallocOpInterface> DialectConversion
    for Bufferizer<A, D>
{
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn BufferizableOpInterface>(Operation::get_op_dyn(op, ctx).as_ref())
            || Operation::get_op::<FuncOp>(op, ctx).is_some()
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        // Handle FuncOp: update function type signature and entry block arg types.
        if let Some(func_op) = Operation::get_op::<FuncOp>(op, ctx) {
            return lower_func_op_to_llvm(&func_op, ctx);
        }
        let op_dyn = Operation::get_op_dyn(op, ctx);
        let op_iface = op_cast::<dyn BufferizableOpInterface>(op_dyn.as_ref())
            .expect("Failed to cast op to BufferizableOpInterface");
        // If any operand aliases to a result, we may need to create a new buffer for that operand.
        let aliases = op_iface.get_operand_result_aliases(ctx);

        let operands = op.deref(ctx).operands_as_uses().collect::<Vec<_>>();
        for opd in operands {
            let aliasing_results = Alias::get_aliases_for_operand(&aliases, opd);
            // If this doesn't alias with a result or is known to be safe to bufferize in-place:
            if aliasing_results.is_empty() || self.in_place_bufferizable_operands.contains(&opd) {
                continue;
            }
            // Create a new buffer for the operand.
            let opd_ty = opd.get_type(ctx);
            let ranked_memref_ty: TypePtr<RankedMemrefType> = TypePtr::from_ptr(opd_ty, ctx)?;
            let dynamic_sizes =
                if let Some(dynamic_sizes) = op_iface.get_dynamic_dimensions(ctx, opd) {
                    dynamic_sizes
                } else {
                    // If dynamic dimensions are not provided by the op, use `memref.dim` to query them (less efficient).
                    let mut dynamic_sizes = Vec::new();
                    let shape = {
                        let ranked_memref_ty = ranked_memref_ty.deref(ctx);
                        ranked_memref_ty.shape().clone()
                    };
                    for (dim_idx, dim) in shape.iter().enumerate() {
                        if matches!(dim, Dimension::Static(_)) {
                            continue;
                        }
                        let dim_const = IndexConstantOp::new(ctx, dim_idx);
                        rewriter.append_op(ctx, dim_const);
                        let dim_size = DimOp::new(ctx, opd.get_def(ctx), dim_const.get_result(ctx))
                            .get_result(ctx);
                        dynamic_sizes.push(dim_size);
                    }
                    dynamic_sizes
                };
            let alloc_op = A::try_new(ctx, ranked_memref_ty, dynamic_sizes)?;
            let new_buffer = alloc_op.get_result(ctx);
            rewriter.append_op(ctx, alloc_op);

            // Copy the operand buffer to the new buffer.
            let copy_op = CopyOp::new(ctx, new_buffer, opd.get_def(ctx));
            rewriter.append_op(ctx, copy_op);
            // Replace the operand with the new buffer.
            Operation::replace_operand(op, ctx, opd.opd_idx, new_buffer);
        }

        // Rewrite the op to use memref semantics.
        op_iface.rewrite(ctx, rewriter, _operands_info)
    }

    fn can_convert_type(&self, ctx: &Context, ty: Ptr<TypeObj>) -> bool {
        type_impls::<dyn ToMemrefType>(&**ty.deref(ctx))
    }

    fn convert_type(&mut self, ctx: &mut Context, ty: Ptr<TypeObj>) -> Result<Ptr<TypeObj>> {
        let to_memref_ty = type_cast::<dyn ToMemrefType>(&**ty.deref(ctx)).map(|t| t.converter());
        if let Some(to_memref_ty) = to_memref_ty {
            to_memref_ty(ty, ctx)
        } else {
            Ok(ty)
        }
    }
}

/// Bufferize `op` and its nested ops
///
/// Bufferization happens in three steps:
/// 1. Compute liveness for all tensor values.
/// 2. For each op, if an operand is aliasing with a result:
///    (a): If the operand is live after the op: Create a new buffer,
///    and, if necessary, copy the operand buffer to it.
///    Replace the operand with the new buffer.
///    (b): If the operand is not live after the op: No action.
/// 3. Rewrite the IR using dialect conversion, which invokes [BufferizableOpInterface::rewrite].
pub fn bufferize<A: MemrefAllocOpInterface, D: MemrefDeallocOpInterface>(
    op: Ptr<Operation>,
    ctx: &mut Context,
) -> Result<()> {
    struct InPlaceBufferizationAnalysis {
        liveness: Liveness<LivenessTq>,
        in_place_bufferizable_operands: FxHashSet<Use<Value>>,
    }

    fn analyze_op(ctx: &Context, state: &mut InPlaceBufferizationAnalysis, node: IRNode) {
        let IRNode::Operation(op) = node else {
            return;
        };

        let op_dyn = Operation::get_op_dyn(op, ctx);
        let Some(op_iface) = op_cast::<dyn BufferizableOpInterface>(op_dyn.as_ref()) else {
            return;
        };

        let aliases = op_iface.get_operand_result_aliases(ctx);
        if aliases.is_empty() {
            return;
        }

        for opd in op.deref(ctx).operands_as_uses() {
            let opd_ty = opd.get_type(ctx);
            if !type_impls::<dyn ToMemrefType>(&**opd_ty.deref(ctx)) {
                continue;
            }

            let aliasing_results = Alias::get_aliases_for_operand(&aliases, opd);
            if aliasing_results.is_empty()
                || Alias::operand_aliases_with_multiple_results(&aliases, opd)
                || Alias::operand_aliases_with_other_operands(&aliases, opd)
            {
                continue;
            }

            if !state.liveness.is_live_at_point(
                ctx,
                opd.get_def(ctx),
                OpInsertionPoint::AfterOperation(op),
            ) {
                state.in_place_bufferizable_operands.insert(opd);
            }
        }
    }

    let mut analysis = InPlaceBufferizationAnalysis {
        liveness: Liveness::<LivenessTq>::default(),
        in_place_bufferizable_operands: FxHashSet::default(),
    };

    walkers::uninterruptible::immutable::walk_op(
        ctx,
        &mut analysis,
        &walkers::WALKCONFIG_PREORDER_FORWARD,
        op,
        analyze_op,
    );

    let mut bufferizer = Bufferizer::<A, D> {
        alloc_op: std::marker::PhantomData,
        dealloc_op: std::marker::PhantomData,
        in_place_bufferizable_operands: analysis.in_place_bufferizable_operands,
    };
    apply_dialect_conversion(ctx, &mut bufferizer, op)
}
