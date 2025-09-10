using Test
using DataFrames
using CSV
using PedigreeTools

@testset "PedigreeTools" begin
    # Data loading tests
    tmpcsv = tempname() * ".csv"
    CSV.write(tmpcsv, DataFrame(ID=["1","2"], sire=["0","1"], dam=["0","0"]))
    dfcsv = load_pedigree(tmpcsv)
    @test nrow(dfcsv) == 2

    # Conversion and restoration
    df = DataFrame(ID=["A","B","C"],
                   sire=["0","A","A"],
                   dam=["0","B","B"],
                   sex=["F","M","F"],
                   birthdate=["2020-01-01","2021-02-01","2022-03-01"],
                   breed=["angus","angus","hereford"])
    df_conv, mapper = convert_pedigree!(copy(df))
    @test eltype(df_conv.ID) <: Integer
    df_rest = restore_pedigree(df_conv, mapper)
    @test Set(skipmissing(df_rest.ID)) == Set(df.ID)

    # Validation
    result = validate_pedigree(df_conv)
    @test result[:has_cycles] == false

    # Lineage extraction
    lin = extract_lineage(df_conv, df_conv.ID[3], generations=2)
    @test size(lin,1) >= 3
end

