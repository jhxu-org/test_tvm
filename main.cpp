#include <topi/broadcast.h>
#include <topi/generic/injective.h>
#include <tvm/driver/driver_api.h>
#include <tvm/ir/module.h>
#include <tvm/node/structural_equal.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/op_strategy.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include "incubator-tvm/3rdparty/dlpack/include/dlpack/dlpack.h"

using namespace tvm;
using namespace tvm::te;

TVM_REGISTER_GLOBAL("test.seq.strategy")
.set_body_typed([](const Attrs& attrs, const Array<te::Tensor>& inputs, const Type& out_type,
                   const Target& target) {
    relay::FTVMCompute fcompute = [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                                     const Type& out_type) -> Array<te::Tensor> {
        CHECK_EQ(inputs.size(), 2U);
        return {topi::add(inputs[0], inputs[1])};
    };
    relay::FTVMSchedule fschedule = [](const Attrs& attrs, const Array<te::Tensor>& outs,
                                       const Target& target) {
        With<Target> target_scope(target);
        return topi::generic::schedule_injective(target, outs);
    };

    auto n = make_object<relay::OpStrategyNode>();
    auto strategy = relay::OpStrategy(std::move(n));
    strategy.AddImplementation(fcompute, fschedule, "test.strategy", 10);
    return strategy;
});



int main() {
    using namespace tvm;
    using namespace tvm::te;
    auto n = var("n");
    Array<PrimExpr> shape;
    shape.push_back(n);

    auto A = placeholder(shape, DataType::Float(32), "A");
    auto B = placeholder(shape, DataType::Float(32), "B");

    auto C = compute(
            A->shape, [&A, &B](PrimExpr i) { return A[i] + B[i]; }, "C");

    auto s = create_schedule({C->op});

    auto cAxis = C->op.as<ComputeOpNode>()->axis;

    IterVar bx, tx;
    s[C].split(cAxis[0], 64, &bx, &tx);

    auto args = Array<Tensor>({A, B, C});
    std::unordered_map<Tensor, Buffer> binds;

    auto target = target::llvm();

    auto lowered = lower(s, args, "func", binds);
    std::cout<<AsText(lowered,false)<<std::endl;
//    auto module = build(lowered, target, Target());
//
//    auto mali_target = Target::Create("opencl -model=Mali-T860MP4@800Mhz -device=mali");
//    CHECK_EQ(
//            mali_target->str(),
//            "opencl -keys=mali,opencl,gpu -device=mali -max_num_threads=256 -model=Mali-T860MP4@800Mhz");
}
