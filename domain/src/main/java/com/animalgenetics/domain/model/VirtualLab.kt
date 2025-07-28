package com.animalgenetics.domain.model

// Using placeholder 'Any' for complex types for now
typealias Genotype = Any
typealias Trait = Any

interface GeneticsSimulation {
    fun simulate(
        parent1Genotype: Genotype,
        parent2Genotype: Genotype,
        traits: List<Trait>
    ): SimulationResult
}

data class SimulationResult(
    val punnettSquare: Map<Pair<Int, Int>, Genotype>,
    val offspringGenotypes: List<Genotype>,
    val phenotypeRatios: Map<String, Float>,
    val genotypeRatios: Map<String, Float>
)

data class ExperimentConfig(
    val type: String,
    val parameters: Map<String, String>
)

sealed class ExperimentState {
    data class Ready(val config: ExperimentConfig) : ExperimentState()
    data class InProgress(val progress: Float) : ExperimentState()
    data class Completed(val results: SimulationResult) : ExperimentState()
    data class Error(val message: String) : ExperimentState()
}

data class ExperimentAction(
    val action: String,
    val data: String
) {
    companion object {
        fun fromJson(action: String, data: String): ExperimentAction {
            // In a real implementation, this would parse the JSON data
            return ExperimentAction(action, data)
        }
    }
}

// Annotation for methods callable from Unity
annotation class UnityCallable
