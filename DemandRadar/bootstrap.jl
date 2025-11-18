(pwd() != @__DIR__) && cd(@__DIR__) # allow starting app from bin/ dir

using DemandRadar
const UserApp = DemandRadar
DemandRadar.main()
