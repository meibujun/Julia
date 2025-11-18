module CreateTableJobs

using SearchLight.Migration

function up()
  create_table(:jobs) do
    [
      pk()
      column(:title, :string)
      column(:company, :string)
      column(:industry, :string)
      column(:location, :string)
      column(:description, :text)
      column(:post_date, :date)
    ]
  end
end

function down()
  drop_table(:jobs)
end

end