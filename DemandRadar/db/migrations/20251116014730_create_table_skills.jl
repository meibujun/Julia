module CreateTableSkills

using SearchLight.Migration

function up()
  create_table(:skills) do
    [
      pk()
      column(:skill_name, :string)
      column(:category, :string)
    ]
  end
end

function down()
  drop_table(:skills)
end

end