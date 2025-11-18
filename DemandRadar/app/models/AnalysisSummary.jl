using SearchLight, Dates

@kwdef mutable struct AnalysisSummary <: AbstractModel
  id::DbId = DbId()
  industry::String = ""
  skill_id::DbId = DbId()
  score::Float64 = 0.0
  month::Date = Dates.now()
  count::Int = 0
end