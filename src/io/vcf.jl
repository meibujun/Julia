# VCF file format I/O

module VcfIO

using DataFrames
using CSV
using ProgressMeter
using ..CoreTypes

export read_vcf
