# PLINK file format I/O

module PlinkIO

using DataFrames
using CSV
using ProgressMeter
using ..CoreTypes

export read_plink
