//! Test conversions of memref operations to Memref -> CF -> LLVM dialect.

use pliron::{
    builtin::ops::ModuleOp,
    combine::Parser,
    context::Context,
    input_error_noloc,
    irbuild::dialect_conversion::apply_dialect_conversion,
    irfmt::parsers::spaced,
    location,
    op::verify_op,
    operation::Operation,
    parsable::{self, state_stream_from_iterator},
    printable::Printable,
    result::ExpectOk,
};

use pliron_common_dialects::cf::to_llvm::CFToLLVM;
use pliron_llvm::llvm_sys::{core::LLVMContext, lljit::LLVMLLJIT, target::initialize_native};

use pliron_tensor::{
    memref::conversions::MemrefToCF,
    tensor::{conversions::TensorToMemref, runtime_utils::TensorDesciptor},
};

use crate::common::init_env_logger;

mod common;

#[test]
fn test_tensor_to_memref_conversion() {
    init_env_logger();
    let ctx = &mut Context::new();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_generate_add: llvm.func <builtin.integer i64 (builtin.integer i64, builtin.integer i64) variadic = false> [] {
                  ^entry(i_res: builtin.integer i64, j_res: builtin.integer i64):
                    input1 = tensor.generate : tensor.ranked<16x16:builtin.integer i64> {
                      ^entry(i_1 : index.index, j_1 : index.index):
                        i_int_1 = index.to_integer i_1 to builtin.integer i64;
                        j_int_1 = index.to_integer j_1 to builtin.integer i64;
                        sum_1 = llvm.add i_int_1, j_int_1 <{nsw = false, nuw = false}> : builtin.integer i64;
                        memref.yield sum_1
                    };
                    input2 = tensor.generate : tensor.ranked<16x16:builtin.integer i64> {
                      ^entry(i_2 : index.index, j_2 : index.index):
                        i_int_2 = index.to_integer i_2 to builtin.integer i64;
                        j_int_2 = index.to_integer j_2 to builtin.integer i64;
                        sum_2 = llvm.add i_int_2, j_int_2 <{nsw = false, nuw = false}> : builtin.integer i64;
                        memref.yield sum_2
                    };
                    res_tensor = tensor.add input1, input2 : tensor.ranked<16x16:builtin.integer i64>;
                    i_res_index = index.from_integer i_res : index.index;
                    j_res_index = index.from_integer j_res : index.index;
                    res = tensor.extract res_tensor[i_res_index, j_res_index]: builtin.integer i64;
                    llvm.return res
                }
            }
            "#;

    let state_stream = state_stream_from_iterator(
        input_ir.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );
    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));

    let parsed_op = parsed.expect_ok(ctx);
    let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
    log::debug!("pliron module parsed {}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut TensorToMemref, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    // Let's try and execute this function
    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_generate_add")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    let f = unsafe { std::mem::transmute::<u64, fn(i64, i64) -> i64>(symbol_addr) };

    for i in 0..16 {
        for j in 0..16 {
            let result = f(i, j);
            assert_eq!(result, ((i + j) * 2));
        }
    }
}

#[test]
fn test_int_tensor_from_rust() {
    init_env_logger();
    let ctx = &mut Context::default();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_tensor_add: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.add arg1, arg2 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                }
            }
            "#;

    let state_stream = state_stream_from_iterator(
        input_ir.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );
    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));

    let parsed_op = parsed.expect_ok(ctx);
    let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
    log::debug!("pliron module parsed {}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut TensorToMemref, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    // Let's try and execute this function
    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tensor_add")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let t1 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].as_ptr() as *const u8,
    );
    let t2 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        [16u64, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1].as_ptr() as *const u8,
    );

    // We build the result descriptor to build the result IR descriptor, where the executed
    // function will write the result descriptor of the addition.
    let res_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(symbol_addr)
    };

    let mut res_ir_descr = res_descr.build_ir_descriptor();

    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    let res_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(res_ir_descr.as_ptr(), 2, std::mem::size_of::<u64>())
    };

    let res_slice = unsafe {
        std::slice::from_raw_parts(
            res_tensor_descr.aligned_ptr() as *const u64,
            res_tensor_descr.num_elements(),
        )
    };

    assert_eq!(res_slice, &[17; 16]);
}

#[test]
fn test_matmul_all_statics_from_rust() {
    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_tensor_matmul: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.matmul arg1, arg2 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                }
            }
            "#;
    test_int_tensor_matmul_from_rust(input_ir);
}

#[test]
fn test_matmul_inner_dynamic_from_rust() {
    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_tensor_matmul: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x?:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<?x4:builtin.integer i64>;
                    res = tensor.matmul arg1, arg2 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                }
            }
            "#;
    test_int_tensor_matmul_from_rust(input_ir);
}

#[test]
fn test_matmul_all_dynamic_from_rust() {
    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_tensor_matmul: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
                    arg1 = llvm.load arg1_p : tensor.ranked<?x?:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<?x?:builtin.integer i64>;
                    res = tensor.matmul arg1, arg2 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                }
            }
            "#;
    test_int_tensor_matmul_from_rust(input_ir);
}

fn test_int_tensor_matmul_from_rust(input_ir: &str) {
    init_env_logger();
    let ctx = &mut Context::default();

    let state_stream = state_stream_from_iterator(
        input_ir.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );
    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));

    let parsed_op = parsed.expect_ok(ctx);
    let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
    log::debug!("pliron module parsed {}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut TensorToMemref, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tensor_matmul")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let t1 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        [1u64, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].as_ptr() as *const u8,
    );
    let t2 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].as_ptr() as *const u8,
    );

    let res_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(symbol_addr)
    };

    let mut res_ir_descr = res_descr.build_ir_descriptor();

    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    let res_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(res_ir_descr.as_ptr(), 2, std::mem::size_of::<u64>())
    };

    let res_slice = unsafe {
        std::slice::from_raw_parts(
            res_tensor_descr.aligned_ptr() as *const u64,
            res_tensor_descr.num_elements(),
        )
    };

    assert_eq!(
        res_slice,
        &[
            28u64, 32, 36, 40, 28, 32, 36, 40, 28, 32, 36, 40, 28, 32, 36, 40
        ]
    );
}

#[test]
fn test_float_tensor_from_rust() {
    init_env_logger();
    let ctx = &mut Context::default();

    let input_ir = r#"
      builtin.module @test_module {
        ^entry():
        llvm.func @test_tensor_add_float: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
          ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
          arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.fp64>;
          arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.fp64>;
          res = tensor.add arg1, arg2 : tensor.ranked<4x4:builtin.fp64>;
          llvm.store *res_p <- res;
          llvm.return
        }
      }
      "#;

    let state_stream = state_stream_from_iterator(
        input_ir.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );
    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));

    let parsed_op = parsed.expect_ok(ctx);
    let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
    log::debug!("pliron module parsed {}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut TensorToMemref, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tensor_add_float")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let t1 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<f64>(),
        [
            1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ]
        .as_ptr() as *const u8,
    );
    let t2 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<f64>(),
        [
            16.0f64, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
            1.0,
        ]
        .as_ptr() as *const u8,
    );

    let res_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<f64>(),
        std::ptr::null::<u8>(),
    );

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(symbol_addr)
    };

    let mut res_ir_descr = res_descr.build_ir_descriptor();

    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    let res_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(res_ir_descr.as_ptr(), 2, std::mem::size_of::<f64>())
    };

    let res_slice = unsafe {
        std::slice::from_raw_parts(
            res_tensor_descr.aligned_ptr() as *const f64,
            res_tensor_descr.num_elements(),
        )
    };

    assert_eq!(res_slice, &[17.0; 16]);
}

#[test]
fn test_float_tensor_all_binary_ops_from_rust() {
    init_env_logger();
    let ctx = &mut Context::default();

    let input_ir = r#"
      builtin.module @test_module {
        ^entry():
        llvm.func @test_tensor_all_binops_float: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
          ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
          arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.fp64>;
          arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.fp64>;
          zero = tensor.sub arg2, arg2 : tensor.ranked<4x4:builtin.fp64>;
          sum = tensor.add arg1, arg2 : tensor.ranked<4x4:builtin.fp64>;
          sum_norm = tensor.add sum, zero : tensor.ranked<4x4:builtin.fp64>;
          prod = tensor.mul sum_norm, arg2 : tensor.ranked<4x4:builtin.fp64>;
          res = tensor.div prod, arg1 : tensor.ranked<4x4:builtin.fp64>;
          llvm.store *res_p <- res;
          llvm.return
        }
      }
      "#;

    let state_stream = state_stream_from_iterator(
        input_ir.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );
    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));

    let parsed_op = parsed.expect_ok(ctx);
    let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
    log::debug!("pliron module parsed {}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut TensorToMemref, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tensor_all_binops_float")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let lhs_data = [
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let rhs_data = [
        16.0f64, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ];

    let t1 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<f64>(),
        lhs_data.as_ptr() as *const u8,
    );
    let t2 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<f64>(),
        rhs_data.as_ptr() as *const u8,
    );

    let res_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<f64>(),
        std::ptr::null::<u8>(),
    );

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(symbol_addr)
    };

    let mut res_ir_descr = res_descr.build_ir_descriptor();

    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    let res_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(res_ir_descr.as_ptr(), 2, std::mem::size_of::<f64>())
    };

    let res_slice = unsafe {
        std::slice::from_raw_parts(
            res_tensor_descr.aligned_ptr() as *const f64,
            res_tensor_descr.num_elements(),
        )
    };

    for ((&a, &b), &c) in lhs_data.iter().zip(rhs_data.iter()).zip(res_slice.iter()) {
        let expected = ((a + b) * b) / a;
        assert!((c - expected).abs() < 1e-12);
    }
}

/// Test that `tensor.extract_slice` is correctly lowered to `memref.extract_slice`
/// by the TensorToMemref conversion pass.
#[test]
fn test_extract_slice_tensor_to_memref() {
    init_env_logger();
    let ctx = &mut Context::new();

    let input_ir = r#"
                builtin.module @test_module {
                    ^entry():
                        llvm.func @test_extract_slice: llvm.func <llvm.void (tensor.ranked<10x20:builtin.integer i64>) variadic = false> [] {
                            ^entry(src : tensor.ranked<10x20:builtin.integer i64>):
                                slice = tensor.extract_slice src [0, 2] [5, 10] [1, 2] : tensor.ranked<5x10:builtin.integer i64>;
                                llvm.return
                        }
                }
        "#;

    let state_stream = state_stream_from_iterator(
        input_ir.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );
    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));
    let parsed_op = parsed.expect_ok(ctx);
    let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut TensorToMemref, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let printed = format!("{}", module_op.disp(ctx));
    assert!(
        !printed.contains("tensor.extract_slice"),
        "tensor.extract_slice should have been lowered"
    );
    assert!(
        printed.contains("memref.extract_slice"),
        "memref.extract_slice should appear after lowering"
    );
    assert!(
        printed.contains(" <- "),
        "destination-style memref.extract_slice syntax should be present"
    );
}

/// Test that `tensor.insert_slice` is lowered to memref / cf ops without leaving
/// tensor dialect slice insertion behind.
#[test]
fn test_insert_slice_tensor_to_memref() {
    init_env_logger();
    let ctx = &mut Context::new();

    let input_ir = r#"
                builtin.module @test_module {
                    ^entry():
                        llvm.func @test_insert_slice: llvm.func <llvm.void (tensor.ranked<5x10:builtin.integer i64>, tensor.ranked<10x20:builtin.integer i64>) variadic = false> [] {
                            ^entry(src : tensor.ranked<5x10:builtin.integer i64>, dst : tensor.ranked<10x20:builtin.integer i64>):
                                updated = tensor.insert_slice src into dst [0, 2] [5, 10] [1, 2] : tensor.ranked<10x20:builtin.integer i64>;
                                llvm.return
                        }
                }
        "#;

    let state_stream = state_stream_from_iterator(
        input_ir.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );
    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));
    let parsed_op = parsed.expect_ok(ctx);
    let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut TensorToMemref, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let printed = format!("{}", module_op.disp(ctx));
    assert!(
        !printed.contains("tensor.insert_slice"),
        "tensor.insert_slice should have been lowered"
    );
    assert!(
        printed.contains("memref.insert_slice"),
        "memref.insert_slice should appear after lowering"
    );
    assert!(
        printed.contains(" <- "),
        "destination-style memref.insert_slice syntax should be present"
    );
    assert!(
        printed.contains(" into "),
        "memref.insert_slice syntax should mention the destination memref"
    );
}
