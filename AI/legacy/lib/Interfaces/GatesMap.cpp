/* This code and any associated documentation is provided "as is"
Copyright 2025 Munich Quantum Software Stack Project
Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at
https://github.com/Munich-Quantum-Software-Stack/passes/blob/develop/LICENSE
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------------------
  author Martin Letras
  date   February 2025
  version 1.0
  brief
    Definition of map used to insert quantum gates into a MLIR module. It
receives as input a tag that identifies the quantum gate, the list of arguments,
control and target qubits.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#include "Interfaces/Constants.hpp"
#include "Interfaces/QASMToQuake.hpp"
#include "Support/mlir_utils.hpp"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Support/Plugin.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mqss::support::quakeDialect;

/*
Suggested Missing Gates (Only Suggest, Do Not Implement)
Based on common QASM/OpenQASM3 gates and Quake dialect:
    ms: Mølmer–Sørensen gate for ion traps.
    fswap: Fermionic SWAP.
    givens: Givens rotation for chemistry sims.
    barrier: For compilation hints (no-op).
 */

void mqss::interfaces::insertQASMGateIntoQuakeModule(
    std::string gateId, OpBuilder &builder, Location loc,
    std::vector<Value> vecParams, std::vector<Value> vecControls,
    std::vector<Value> vecTargets, bool adj) {
  std::ranges::transform(gateId, gateId.begin(),
                         [](unsigned char c) { return std::tolower(c); });
  ValueRange params(vecParams);
  ValueRange controls(vecControls);
  ValueRange targets(vecTargets);
  Value plusHalfPi = createFloatValue(builder, loc, PI_2);
  Value minusHalfPi = createFloatValue(builder, loc, -PI_2);
#ifdef DEBUG
  std::cout << "gate " << gateId << std::endl;
  std::cout << "controls size " << controls.size() << std::endl;
  std::cout << "target size " << targets.size() << std::endl;
  std::cout << "params size " << params.size() << std::endl;
#endif
  static const std::unordered_map<std::string, std::function<void()>> gateMap =
      {{"id",
        [&] {
          /* Identity does nothing. */
        }},
       {"gphase",
        [&] {
          /* Global phase does not affect measurement. */
        }},
       {"x",
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 "ill-formed x gate");
          builder.create<quake::XOp>(loc, false, ValueRange{}, ValueRange{},
                                     targets);
        }},
       {"y",
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 "ill-formed y gate");
          builder.create<quake::YOp>(loc, false, ValueRange{}, ValueRange{},
                                     targets);
        }},
       {"z",
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 "ill-formed z gate");
          builder.create<quake::ZOp>(loc, false, ValueRange{}, ValueRange{},
                                     targets);
        }},
       {"h",
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 "ill-formed h gate");
          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     targets);
        }},
       {"s",
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 !adj && "ill-formed s gate");
          builder.create<quake::SOp>(loc, false, ValueRange{}, ValueRange{},
                                     targets);
        }},
       {"sdg",
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 adj && "ill-formed sdg gate");
          builder.create<quake::SOp>(loc, true, ValueRange{}, ValueRange{},
                                     targets);
        }},
       {"t",
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 !adj && "ill-formed t gate");
          builder.create<quake::TOp>(loc, false, ValueRange{}, ValueRange{},
                                     targets);
        }},
       {"tdg",
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 adj && "ill-formed tdg gate");
          builder.create<quake::TOp>(loc, true, ValueRange{}, ValueRange{},
                                     targets);
        }},
       {"rx",
        [&] {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed rx gate");
          builder.create<quake::RxOp>(loc, false, params, ValueRange{},
                                      targets);
        }},
       {"ry",
        [&] {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed ry gate");
          builder.create<quake::RyOp>(loc, false, params, ValueRange{},
                                      targets);
        }},
       {"rz",
        [&] {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed rz gate");
          builder.create<quake::RzOp>(loc, false, params, ValueRange{},
                                      targets);
        }},
       {"r",
        [&] {
          assert(params.size() == 2 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed r gate");
          builder.create<quake::PhasedRxOp>(loc, false, params, ValueRange{},
                                            targets);
        }},
       {"p",
        [&] {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed p gate");
          builder.create<quake::R1Op>(loc, false, params, ValueRange{},
                                      targets);
        }},
       {"phase",
        [&] {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed phase gate");
          builder.create<quake::R1Op>(loc, false, params, ValueRange{},
                                      targets);
        }},
       {"sx",
        // Sx = Rx(π/2)
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 !adj && "ill-formed sx gate");
          builder.create<quake::RxOp>(loc, false, plusHalfPi, ValueRange{},
                                      targets);
        }},
       {"sxdg",
        // Sxdg = Rx(-π/2)
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 1 &&
                 adj && "ill-formed sxdg gate");
          builder.create<quake::RxOp>(loc, false, minusHalfPi, ValueRange{},
                                      targets);
        }},
       {"u1",
        [&] {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed u1 gate");
          builder.create<quake::R1Op>(loc, false, params, ValueRange{},
                                      targets);
        }},
       {"u2",
        [&] {
          assert(params.size() == 2 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed u2 gate");
          builder.create<quake::U2Op>(loc, adj, params, controls, targets);
        }},
       {"u3",
        [&] {
          assert(params.size() == 3 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed u3 gate");
          builder.create<quake::U3Op>(loc, adj, params, controls, targets);
        }},
       {"u",
        [&] {
          assert(params.size() == 3 && controls.empty() && !adj &&
                 targets.size() == 1 && "ill-formed u gate");
          builder.create<quake::U3Op>(loc, adj, params, controls, targets);
        }},
       {"cx",
        [&] {
          assert(params.empty() && controls.size() == 1 &&
                 targets.size() == 1 && "ill-formed cx gate");
          builder.create<quake::XOp>(loc, false, ValueRange{}, controls,
                                     targets);
        }},
       {"cy",
        [&] {
          assert(params.empty() && controls.size() == 1 &&
                 targets.size() == 1 && "ill-formed cy gate");
          builder.create<quake::YOp>(loc, false, ValueRange{}, controls,
                                     targets);
        }},
       {"cz",
        [&] {
          assert(params.empty() && controls.size() == 1 &&
                 targets.size() == 1 && "ill-formed cz gate");
          builder.create<quake::ZOp>(loc, false, ValueRange{}, controls,
                                     targets);
        }},
       {"ch",
        [&] {
          assert(params.empty() && controls.size() == 1 &&
                 targets.size() == 1 && "ill-formed ch gate");
          builder.create<quake::HOp>(loc, false, ValueRange{}, controls,
                                     targets);
        }},
       {"cs",
        [&] {
          assert(params.empty() && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cs gate");
          builder.create<quake::SOp>(loc, false, ValueRange{}, controls,
                                     targets);
        }},
       {"csdg",
        [&] {
          assert(params.empty() && controls.size() == 1 && adj &&
                 targets.size() == 1 && "ill-formed csdg gate");
          builder.create<quake::SOp>(loc, true, ValueRange{}, controls,
                                     targets);
        }},
       {"ct",
        [&] {
          assert(params.empty() && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed ct gate");
          builder.create<quake::TOp>(loc, false, ValueRange{}, controls,
                                     targets);
        }},
       {"ctdg",
        [&] {
          assert(params.empty() && controls.size() == 1 && adj &&
                 targets.size() == 1 && "ill-formed ctdg gate");
          builder.create<quake::TOp>(loc, true, ValueRange{}, controls,
                                     targets);
        }},
       {"crx",
        [&] {
          assert(params.size() == 1 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed crx gate");
          builder.create<quake::RxOp>(loc, false, params, controls, targets);
        }},
       {"cry",
        [&] {
          assert(params.size() == 1 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cry gate");
          builder.create<quake::RyOp>(loc, false, params, controls, targets);
        }},
       {"crz",
        [&] {
          assert(params.size() == 1 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed crz gate");
          builder.create<quake::RzOp>(loc, false, params, controls, targets);
        }},
       {"cr",
        [&] {
          assert(params.size() == 2 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cr gate");
          builder.create<quake::PhasedRxOp>(loc, false, params, controls,
                                            targets);
        }},
       {"cp",
        [&] {
          assert(params.size() == 1 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cp gate");
          builder.create<quake::R1Op>(loc, false, params, controls, targets);
        }},
       {"cphase",
        [&] {
          assert(params.size() == 1 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cphase gate");
          builder.create<quake::R1Op>(loc, false, params, controls, targets);
        }},
       {"cu1",
        [&] {
          assert(params.size() == 1 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cu1 gate");
          builder.create<quake::R1Op>(loc, false, params, controls, targets);
        }},
       {"cu2",
        [&] {
          assert(params.size() == 2 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cu2 gate");
          builder.create<quake::U2Op>(loc, false, params, controls, targets);
        }},
       {"cu3",
        [&] {
          assert(params.size() == 3 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cu3 gate");
          builder.create<quake::U3Op>(loc, false, params, controls, targets);
        }},
       {"cu",
        [&] {
          assert(params.size() == 3 && controls.size() == 1 && !adj &&
                 targets.size() == 1 && "ill-formed cu gate");
          builder.create<quake::U3Op>(loc, false, params, controls, targets);
        }},
       {"swap",
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 2 &&
                 "ill-formed swap gate");
          builder.create<quake::SwapOp>(loc, false, params, controls, targets);
        }},
       {"iswap",
        // iSWAP q1, q2:
        //  S q1
        //  S q2
        //  H q1
        //  Cx q1, q2
        //  Cx q2, q1
        //  H q2
        [&] {
          assert(params.empty() && controls.empty() && !adj &&
                 targets.size() == 2 && "ill-formed iswap gate");
          auto q1 = targets[0];
          auto q2 = targets[1];

          builder.create<quake::SOp>(loc, false, ValueRange{}, ValueRange{},
                                     q1);
          builder.create<quake::SOp>(loc, false, ValueRange{}, ValueRange{},
                                     q2);
          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     q1);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q2, q1);
          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     q2);
        }},
       {"iswapdg",
        [&] {
          // iSWAPdg (q1, q2) {
          //  H q2
          //  Cx q2, q1
          //  Cx q1, q2
          //  H q1
          //  Sdg q1
          //  Sdg q2
          assert(params.empty() && controls.empty() && adj &&
                 targets.size() == 2 && "ill-formed iswapdg gate");
          auto q1 = targets[0];
          auto q2 = targets[1];

          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q2, q1);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     q1);
          builder.create<quake::SOp>(loc, true, ValueRange{}, ValueRange{}, q1);
          builder.create<quake::SOp>(loc, true, ValueRange{}, ValueRange{}, q2);
        }},
       {"ccx",
        [&] {
          assert(params.empty() && controls.size() == 2 &&
                 targets.size() == 1 && "ill-formed ccx gate");
          builder.create<quake::XOp>(loc, false, ValueRange{}, controls,
                                     targets);
        }},
       {"ccy",
        [&] {
          assert(params.empty() && controls.size() == 2 &&
                 targets.size() == 1 && "ill-formed ccy gate");
          builder.create<quake::YOp>(loc, false, ValueRange{}, controls,
                                     targets);
        }},
       {"ccz",
        [&] {
          assert(params.empty() && controls.size() == 2 &&
                 targets.size() == 1 && "ill-formed ccz gate");
          builder.create<quake::ZOp>(loc, false, ValueRange{}, controls,
                                     targets);
        }},
       {"cswap",
        [&] {
          assert(params.empty() && controls.size() == 1 &&
                 targets.size() == 2 && "ill-formed cswap gate");
          builder.create<quake::SwapOp>(loc, false, ValueRange{}, controls,
                                        targets);
        }},
       {"toffoli",
        [&] {
          assert(params.empty() && controls.size() == 2 &&
                 targets.size() == 1 && "ill-formed toffoli gate");
          builder.create<quake::XOp>(loc, false, ValueRange{}, controls,
                                     targets);
        }},
       {"fredkin",
        [&] {
          assert(params.empty() && controls.size() == 1 &&
                 targets.size() == 2 && "ill-formed fredkin gate");
          builder.create<quake::SwapOp>(loc, false, ValueRange{}, controls,
                                        targets);
        }},
       {"rccx",
        // Rccx q1, q2, q3:
        //  Cz q1, q3
        //  H q3
        //  T q3
        //  Cx q2, q3
        //  Tdg q3
        //  Cx q1, q3
        //  T 3
        //  Cx q2, q3
        //  Tdg q3
        //  H q3
        [&] {
          assert(params.empty() && controls.size() == 2 &&
                 targets.size() == 1 && "ill-formed rccx gate");
          auto q1 = controls[0];
          auto q2 = controls[1];
          auto q3 = targets[0];

          builder.create<quake::ZOp>(loc, false, ValueRange{}, q1, q3);
          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     q3);
          builder.create<quake::TOp>(loc, false, ValueRange{}, ValueRange{},
                                     q3);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q2, q3);
          builder.create<quake::TOp>(loc, true, ValueRange{}, ValueRange{}, q3);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q3);
          builder.create<quake::TOp>(loc, false, ValueRange{}, ValueRange{},
                                     q3);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q2, q3);
          builder.create<quake::TOp>(loc, true, ValueRange{}, ValueRange{}, q3);
          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     q3);
        }},
       {"rxx",
        // Rxx(θ) q1, q2:
        //  H q1
        //  H q2
        //  Cx q1, q2
        //  Rz(θ) q2
        //  Cx q1, q2
        //  H q2
        //  H q1
        [&] {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 2 && "ill-formed rxx gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          auto theta = params[0];
          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     q1);
          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::RzOp>(loc, false, theta, ValueRange{}, q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     q2);
          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     q1);
        }},
       {"ryy",
        // Ryy(θ) q1, q2:
        //  Rx(π/2) q1
        //  Rx(π/2) q2
        //  Cx q1, q2
        //  Rz(θ) q2
        //  Cx q1, q2
        //  Rx(-π/2) q2
        //  Rx(-π/2) q1
        [&] {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 2 && "ill-formed ryy gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          auto param1 = plusHalfPi;
          auto param2 = params[0];
          auto param3 = minusHalfPi;

          // Optimally, one would want to pass the targets to the Rx gate in one
          // go like this: builder.create<quake::RxOp>(loc, false, param1,
          // ValueRange{}, targets); But if you do that, it throws the error:
          // 'quake.rx' op failed to verify that the number of targets is equal
          // to 1
          builder.create<quake::RxOp>(loc, false, param1, ValueRange{}, q1);
          builder.create<quake::RxOp>(loc, false, param1, ValueRange{}, q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::RzOp>(loc, false, param2, ValueRange{}, q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::RxOp>(loc, false, param3, ValueRange{}, q2);
          builder.create<quake::RxOp>(loc, false, param3, ValueRange{}, q1);
        }},
       {"rzz",
        // Rzz(θ) q1, q2:
        //  Cx q1, q2
        //  Rz(θ) q2
        //  Cx q1, q2
        [&] {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 2 && "ill-formed rzz gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          auto theta = params[0];
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::RzOp>(loc, false, theta, ValueRange{}, q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
        }},
       {"rzx",
        // Rzx(θ) q1, q2:
        //  H q1
        //  Cx q1, q2
        //  Rx(θ) q2
        //  Cx q1, q2
        //  H q1
        [&] {
          assert(params.size() == 1 && controls.empty() && !adj &&
                 targets.size() == 2 && "ill-formed rzx gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          auto theta = params[0];
          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     q1);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::RxOp>(loc, false, theta, ValueRange{}, q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::HOp>(loc, false, ValueRange{}, ValueRange{},
                                     q1);
        }},
       {"dcx",
        // Dcx q1, q2:
        //  Cx q1, q2
        //  Cx q2, q1
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 2 &&
                 !adj && "ill-formed dcx gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q2, q1);
        }},
       {"ecr",
        // Ecr q1, q2:
        //  S q1
        //  Rx(π/2) q2
        //  Cx q1, q2
        //  X q1
        [&] {
          assert(params.empty() && controls.empty() && targets.size() == 2 &&
                 !adj && "ill-formed ecr gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          auto param1 = plusHalfPi;
          builder.create<quake::SOp>(loc, false, ValueRange{}, ValueRange{},
                                     q1);
          builder.create<quake::RxOp>(loc, false, param1, ValueRange{}, q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, ValueRange{},
                                     q1);
        }},
       {"xx_plus_yy",
        // gate xx_plus_yy(param0,param1) q1,q2 {
        //   rz(param1) q1;
        //   sdg q2;
        //   sx q2;
        //   s q2;
        //   s q1;
        //   cx q2,q1;
        //   ry((-0.5)*param0) q2;
        //   ry((-0.5)*param0) q1;
        //   cx q2,q1;
        //   sdg q1;
        //   sdg q2;
        //   sxdg q2;
        //   s q2;
        //   rz(-param1) q1;
        // }
        [&] {
          assert(params.size() == 2 && controls.empty() &&
                 targets.size() == 2 && !adj && "ill-formed xx_plus_yy gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          auto param1 = params[1];
          auto param2 = plusHalfPi;
          auto param0ValueOpt =
              extractDoubleArgumentValue(params[0].getDefiningOp());
          auto param1ValueOpt =
              extractDoubleArgumentValue(params[1].getDefiningOp());
          if (!param0ValueOpt.has_value() || !param1ValueOpt.has_value()) {
            return;
          }
          auto param3 =
              createFloatValue(builder, loc, -0.5 * param0ValueOpt.value());
          auto param4 = minusHalfPi;
          auto param5 = createFloatValue(builder, loc, -param1ValueOpt.value());
          builder.create<quake::RzOp>(loc, false, param1, ValueRange{}, q1);
          builder.create<quake::SOp>(loc, true, ValueRange{}, ValueRange{}, q2);
          builder.create<quake::RxOp>(loc, false, param2, ValueRange{}, q2);
          builder.create<quake::SOp>(loc, false, ValueRange{}, ValueRange{},
                                     q2);
          builder.create<quake::SOp>(loc, false, ValueRange{}, ValueRange{},
                                     q1);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q2, q1);
          builder.create<quake::RyOp>(loc, false, param3, ValueRange{}, q2);
          builder.create<quake::RyOp>(loc, false, param3, ValueRange{}, q1);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q2, q1);
          builder.create<quake::SOp>(loc, true, ValueRange{}, ValueRange{}, q1);
          builder.create<quake::SOp>(loc, true, ValueRange{}, ValueRange{}, q2);
          builder.create<quake::RxOp>(loc, false, param4, ValueRange{}, q2);
          builder.create<quake::SOp>(loc, false, ValueRange{}, ValueRange{},
                                     q2);
          builder.create<quake::RzOp>(loc, false, param5, ValueRange{}, q1);
        }},
       {"xx_minus_yy",
        // gate xx_minus_yy(param0,param1) q1,q2 {
        //   rz(-param1) q2;
        //   sdg q1;
        //   sx q1;
        //   s q1;
        //   s q2;
        //   cx q1,q2;
        //   ry(0.5*param0) q1;
        //   ry((-0.5)*param0) q2;
        //   cx q1,q2;
        //   sdg q2;
        //   sdg q1;
        //   sxdg q1;
        //   s q1;
        //   rz(param1) q2;
        // }
        [&] {
          assert(params.size() == 2 && controls.empty() &&
                 targets.size() == 2 && !adj && "ill-formed xx_minus_yy gate");
          auto q1 = targets[0];
          auto q2 = targets[1];
          auto param0ValueOpt =
              extractDoubleArgumentValue(params[0].getDefiningOp());
          auto param1ValueOpt =
              extractDoubleArgumentValue(params[1].getDefiningOp());
          if (!param0ValueOpt.has_value() || !param1ValueOpt.has_value()) {
            return;
          }
          auto param1 = createFloatValue(builder, loc, -param1ValueOpt.value());
          auto param2 = plusHalfPi;
          auto param3 =
              createFloatValue(builder, loc, 0.5 * param0ValueOpt.value());
          auto param4 =
              createFloatValue(builder, loc, -0.5 * param0ValueOpt.value());
          auto param5 = minusHalfPi;
          auto param6 = params[1];
          builder.create<quake::RzOp>(loc, false, param1, ValueRange{}, q2);
          builder.create<quake::SOp>(loc, true, ValueRange{}, ValueRange{}, q1);
          builder.create<quake::RxOp>(loc, false, param2, ValueRange{}, q1);
          builder.create<quake::SOp>(loc, false, ValueRange{}, ValueRange{},
                                     q1);
          builder.create<quake::SOp>(loc, false, ValueRange{}, ValueRange{},
                                     q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::RyOp>(loc, false, param3, ValueRange{}, q1);
          builder.create<quake::RyOp>(loc, false, param4, ValueRange{}, q2);
          builder.create<quake::XOp>(loc, false, ValueRange{}, q1, q2);
          builder.create<quake::SOp>(loc, true, ValueRange{}, ValueRange{}, q2);
          builder.create<quake::SOp>(loc, true, ValueRange{}, ValueRange{}, q1);
          builder.create<quake::RxOp>(loc, false, param5, ValueRange{}, q1);
          builder.create<quake::SOp>(loc, false, ValueRange{}, ValueRange{},
                                     q1);
          builder.create<quake::RzOp>(loc, false, param6, ValueRange{}, q2);
        }}};
  if (auto it = gateMap.find(gateId); it != gateMap.end()) {
    it->second();
  } else {
    assert(false && ("Unknown gate: " + gateId + "\n").c_str());
  }
}