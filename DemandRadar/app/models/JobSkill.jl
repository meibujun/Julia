using SearchLight

@kwdef mutable struct JobSkill <: AbstractModel
  id::DbId = DbId()
  job_id::DbId = DbId()
  skill_id::DbId = DbId()
  weight::Float64 = 0.0
end