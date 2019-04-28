local _M = {}

_M.BizhawkDir = "C:/Users/Benjamin/Documents/SNES Shit/BizHawk-2.2.2/"

_M.StateDir = _M.BizhawkDir .. "Lua/SNES/crb/neat-mario-fco/state/"
_M.PoolDir = _M.BizhawkDir .. "Lua/SNES/crb/neat-mario-fco/pool/"

_M.State = {
    "YI2.state", -- Yoshi's Island 2
    "DP1.state", -- Donut Plains 1
}
_M.StateNo = 2

_M.WhichState = _M.State[_M.StateNo]

_M.Records = _M.PoolDir .. "records.csv"

_M.NeatConfig = {
    Filename = _M.StateDir .. _M.State[_M.StateNo],
    Population = 300,
    DeltaDisjoint = 2.0,
    DeltaWeights = 0.4,
    DeltaThreshold = 1.0,
    StaleSpecies = 15,
    MutateConnectionsChance = 0.25,
    PerturbChance = 0.90,
    CrossoverChance = 0.75,
    LinkMutationChance = 2.0,
    NodeMutationChance = 0.50,
    BiasMutationChance = 0.40,
    StepSize = 0.1,
    DisableMutationChance = 0.4,
    EnableMutationChance = 0.2,
    TimeoutConstant = 20,
    MaxNodes = 1000000,
}

_M.ButtonNames = {
    "A",
    "B",
    "X",
    "Y",
    "Up",
    "Down",
    "Left",
    "Right",
}

_M.BoxRadius = 6
_M.InputSize = (_M.BoxRadius * 2 + 1) * (_M.BoxRadius * 2 + 1)

_M.Running = false
_M.Testing = false
_M.Network = false

return _M
