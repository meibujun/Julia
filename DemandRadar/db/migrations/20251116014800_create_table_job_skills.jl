module CreateTableJobSkills

using SearchLight.Migration

function up()
  create_table(:job_skills) do
    [
      pk()
      column(:job_id, :integer)
      column(:skill_id, :integer)
      column(:weight, :float)
    ]
  end
end

function down()
  drop_table(:job_skills)
end

end