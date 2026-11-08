include "mlir/Pass/PassBase.td"

def XXToId : Pass<"XXToId"> {
  let summary = "Remove consecutive X gates";
  let constructor = "mqss::opt::createXXToIdPass()";
}

def YYToId : Pass<"YYToId"> {
  let summary = "Remove consecutive Y gates";
  let constructor = "mqss::opt::createYYToIdPass()";
}

def ZZToId : Pass<"ZZToId"> {
  let summary = "Remove consecutive Z gates";
  let constructor = "mqss::opt::createZZToIdPass()";
}

def SSdgToId : Pass<"SSdgToId"> {
  let summary = "Remove S followed by Sdg";
  let constructor = "mqss::opt::createSSdgToIdPass()";
}

def SdgSToId : Pass<"SdgSToId"> {
  let summary = "Remove Sdg followed by S";
  let constructor = "mqss::opt::createSdgSToIdPass()";
}

def TTdgToId : Pass<"TTdgToId"> {
  let summary = "Remove T followed by Tdg";
  let constructor = "mqss::opt::createTTdgToIdPass()";
}

def TdgTToId : Pass<"TdgTToId"> {
  let summary = "Remove Tdg followed by T";
  let constructor = "mqss::opt::createTdgTToIdPass()";
}

def HHToId : Pass<"HHToId"> {
  let summary = "Remove consecutive H gates";
  let constructor = "mqss::opt::createHHToIdPass()";
}

def CxCxToId : Pass<"CxCxToId"> {
  let summary = "Remove consecutive identical Cx gates.";
  let constructor = "mqss::opt::createCxCxToIdPass()";
}

def CyCyToId : Pass<"CyCyToId"> {
  let summary = "Remove consecutive identical Cy gates.";
  let constructor = "mqss::opt::createCyCyToIdPass()";
}

def CzCzToId : Pass<"CzCzToId"> {
  let summary = "Remove consecutive identical Cz gates.";
  let constructor = "mqss::opt::createCzCzToIdPass()";
}

def ZeroRxToId : Pass<"ZeroRxToId"> {
  let summary = "Remove Rx gates with zero angle";
  let constructor = "mqss::opt::createZeroRxToIdPass()";
}

def ZeroRyToId : Pass<"ZeroRyToId"> {
  let summary = "Remove Ry gates with zero angle";
  let constructor = "mqss::opt::createZeroRyToIdPass()";
}

def ZeroRzToId : Pass<"ZeroRzToId"> {
  let summary = "Remove Rz gates with zero angle";
  let constructor = "mqss::opt::createZeroRzToIdPass()";
}
