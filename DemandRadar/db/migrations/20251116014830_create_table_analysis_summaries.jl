module CreateTableAnalysisSummaries

using SearchLight.Migration

function up()
  create_table(:analysis_summaries) do
    [
      pk()
      column(:industry, :string)
      column(:skill_id, :integer)
      column(:score, :float)
      column(:month, :date)
      column(:count, :integer)
    ]
  end
end

function down()
  drop_table(:analysis_summaries)
end

end