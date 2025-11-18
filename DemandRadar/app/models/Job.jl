using SearchLight, Dates

@kwdef mutable struct Job <: AbstractModel
  id::DbId = DbId()
  title::String = ""
  company::String = ""
  industry::String = ""
  location::String = ""
  description::String = ""
  post_date::Date = Dates.now()
end