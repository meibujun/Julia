using SearchLight

@kwdef mutable struct Skill <: AbstractModel
  id::DbId = DbId()
  skill_name::String = ""
  category::String = ""
end