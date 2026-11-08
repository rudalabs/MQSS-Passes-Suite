/* This code and any associated documentation is provided "as is"

Copyright 2024 Munich Quantum Software Stack Project

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
*************************************************************************
  author Martin Letras
  date   February 2025
  version 1.0
*************************************************************************/

include "mlir/Pass/PassBase.td"

def NormalizeArgAngle : Pass<"NormalizeArgAngle"> {
  let summary =
      "Optimization pass that normalizes the angle of Rx, Ry and Rz rotations";
  let constructor = "mqss::opt::createNormalizeArgAnglePass()";
}

def CxRxToRxCx : Pass<"CxRxToRxCx"> {
  let summary = "Apply commutation pass of pattern Cx-Rx to Rx-Cx";
  let constructor = "mqss::opt::createCxRxToRxCxPass()";
}

def CxXToXCx : Pass<"CxXToXCx"> {
  let summary = "Apply commutation pass to pattern Cx-X to X-Cx";
  let constructor = "mqss::opt::createCxXToXCxPass()";
}

def CxZToZCx : Pass<"CxZToZCx"> {
  let summary = "Apply commutation pass to pattern Cx-Z to Z-Cx";
  let constructor = "mqss::opt::createCxZToZCxPass()";
}

def RxCxToCxRx : Pass<"RxCxToCxRx"> {
  let summary = "Apply commutation pass to pattern Rx-Cx to Cx-Rx";
  let constructor = "mqss::opt::createRxCxToCxRxPass()";
}

def XCxToCxX : Pass<"XCxToCxX"> {
  let summary = "Apply commutation pass to pattern X-Cx to Cx-X";
  let constructor = "mqss::opt::createXCxToCxXPass()";
}

def ZCxToCxZ : Pass<"ZCxToCxZ"> {
  let summary = "Apply commutation pass to pattern Z-Cx to Cx-Z";
  let constructor = "mqss::opt::createZCxToCxZPass()";
}

def CxCxCxToSwap : Pass<"CxCxCxToSwap"> {
  let summary = "Replace CX CX CX pattern by SWAP";
  let constructor = "mqss::opt::createCxCxCxToSwapPass()";
}

def HXHToZ : Pass<"HXHToZ"> {
  let summary =
      " Optimization pass that replaces a pattern composed of H, X, H by Z";
  let constructor = "mqss::opt::createHXHToZPass()";
}

def HZHToX : Pass<"HZHToX"> {
  let summary =
      "Optimization pass that replaces a pattern composed of H, Z, H by X";
  let constructor = "mqss::opt::createHZHToXPass()";
}

def SdgZToS : Pass<"SdgZToS"> {
  let summary = "Optimization pass that replaces a pattern composed of S "
                "adjoint and Z by S";
  let constructor = "mqss::opt::createSdgZToSPass()";
}

def ZSdgToS : Pass<"ZSdgToS"> {
  let summary = "Optimization pass that replaces a pattern composed of S "
                "adjoint and Z by S";
  let constructor = "mqss::opt::createZSdgToSPass()";
}

def SZToSdg : Pass<"SZToSdg"> {
  let summary =
      "Optimization pass that replaces a pattern composed of S and Z by Sdg";
  let constructor = "mqss::opt::createSZToSdgPass()";
}

def ZSToSdg : Pass<"ZSToSdg"> {
  let summary =
      "Optimization pass that replaces a pattern composed of S and Z by Sdg";
  let constructor = "mqss::opt::createZSToSdgPass()";
}

def HCRxHToCrzPass : Pass<"HCrxHToCrz"> {
  let summary = "Decomposition pass of H, Crx, H to Crz";
  let constructor = "mqss::opt::createHCrxHToCrzPass()";
}

def HCRzHToCrxPass : Pass<"HCrzHToCrx"> {
  let summary = "Decomposition pass of H, Crz, H to Crx";
  let constructor = "mqss::opt::createHCrzHToCrxPass()";
}

def HYToYH : Pass<"HYToYH"> {
  let summary = "Transform the pattern HY to YH";
  let constructor = "mqss::opt::createHYToYHPass()";
}

def YHToHY : Pass<"YHToHY"> {
  let summary = "Transform the pattern YH to HY";
  let constructor = "mqss::opt::createYHToHYPass()";
}

def HCzHToCx : Pass<"HCzHToCx"> {
  let summary = "Decomposition pass of H, Cz, H to Cx";
  let constructor = "mqss::opt::createHCzHToCxPass()";
}

def HCxHToCz : Pass<"HCxHToCz"> {
  let summary = "Decomposition pass of H, Cx, H to Cz";
  let constructor = "mqss::opt::createHCxHToCzPass()";
}

def XHZToH : Pass<"XHZToH"> {
  let summary = "Optimization pass that replaces a pattern composed of X, H, Z by H";
  let constructor = "mqss::opt::createXHZToHPass()";
}

def ZHXToH : Pass<"ZHXToH"> {
  let summary = "Optimization pass that replaces a pattern composed of Z, H, X by H";
  let constructor = "mqss::opt::createZHXToHPass()";
}

def TTToS : Pass<"TTToS"> {
  let summary = "Optimization pass that replaces a pattern composed of two T gates with an S gate";
  let constructor = "mqss::opt::createTTToSPass()";
}

def RxRxToRx : Pass<"RxRxToRx"> {
  let summary = "Collapse two consecutive Rx gates into one";
  let constructor = "mqss::opt::createRxRxToRxPass()";
}

def RyRyToRy : Pass<"RyRyToRy"> {
  let summary = "Collapse two consecutive Ry gates into one";
  let constructor = "mqss::opt::createRyRyToRyPass()";
}

def RzRzToRz : Pass<"RzRzToRz"> {
  let summary = "Collapse two consecutive Rz gates into one";
  let constructor = "mqss::opt::createRzRzToRzPass()";
}

def HRxHToRz : Pass<"HRxHToRz"> {
  let summary = "Fold H Rx H to Rz";
  let constructor = "mqss::opt::createHRxHToRzPass()";
}

def HRzHToRx : Pass<"HRzHToRx"> {
  let summary = "Fold H Rz H to Rx";
  let constructor = "mqss::opt::createHRzHToRxPass()";
}

def SdgSdgSdgToS : Pass<"SdgSdgSdgToS"> {
  let summary = "Replace Sdg Sdg Sdg with S";
  let constructor = "mqss::opt::createSdgSdgSdgToSPass()";
}

def SdgSdgToZ : Pass<"SdgSdgToZ"> {
  let summary = "Replace Sdg Sdg with Z";
  let constructor = "mqss::opt::createSdgSdgToZPass()";
}

def SSSToSdg : Pass<"SSSToSdg"> {
  let summary = "Replace S S S with Sdg";
  let constructor = "mqss::opt::createSSSToSdgPass()";
}

def SSToZ : Pass<"SSToZ"> {
  let summary = "Replace S S with Z";
  let constructor = "mqss::opt::createSSToZPass()";
}

def HXToZH : Pass<"HXToZH"> {
  let summary = "Switch the pattern H and X to Z and H";
  let constructor = "mqss::opt::createHXToZHPass()";
}

def HZToXH : Pass<"HZToXH"> {
  let summary = "Switch the pattern H and Z to X and H";
  let constructor = "mqss::opt::createHZToXHPass()";
}

def XHToHZ : Pass<"XHToHZ"> {
  let summary = "Switch the pattern X and H to H and Z";
  let constructor = "mqss::opt::createXHToHZPass()";
}

def ZHToHX : Pass<"ZHToHX"> {
  let summary = "Switch the pattern Z and H to H and X";
  let constructor = "mqss::opt::createZHToHXPass()";
}

