package com.animalgenetics.feature.virtuallab.simulation

import com.animalgenetics.domain.model.GeneticsSimulation
import com.animalgenetics.domain.model.SimulationResult

// --- Placeholders for complex domain models ---
data class Genotype(val alleles: String) {
    // e.g., "Aa", "Bb", "AABB"
    fun getAllelesForTrait(trait: Trait): String {
        return alleles.substring(trait.geneIndex, trait.geneIndex + 2)
    }
}

data class Trait(val name: String, val dominantAllele: Char, val recessiveAllele: Char, val geneIndex: Int)

// --- Simulation Implementation ---

class PunnettSquareSimulation : GeneticsSimulation {

    override fun simulate(
        parent1Genotype: com.animalgenetics.domain.model.Genotype,
        parent2Genotype: com.animalgenetics.domain.model.Genotype,
        traits: List<com.animalgenetics.domain.model.Trait>
    ): SimulationResult {
        // This is a simplified example for a single trait (monohybrid cross)
        val trait = (traits.first() as Trait)
        val p1 = (parent1Genotype as Genotype).getAllelesForTrait(trait)
        val p2 = (parent2Genotype as Genotype).getAllelesForTrait(trait)

        val gametes1 = listOf(p1[0], p1[1])
        val gametes2 = listOf(p2[0], p2[1])

        val offspringGenotypes = mutableListOf<Genotype>()
        val punnettSquare = mutableMapOf<Pair<Int, Int>, com.animalgenetics.domain.model.Genotype>()

        gametes1.forEachIndexed { i, g1 ->
            gametes2.forEachIndexed { j, g2 ->
                val offspringAlleles = String(charArrayOf(g1, g2).sortedArray())
                val offspring = Genotype(offspringAlleles)
                offspringGenotypes.add(offspring)
                punnettSquare[i to j] = offspring
            }
        }

        return SimulationResult(
            punnettSquare = punnettSquare,
            offspringGenotypes = offspringGenotypes,
            phenotypeRatios = calculatePhenotypeRatios(offspringGenotypes, traits),
            genotypeRatios = calculateGenotypeRatios(offspringGenotypes)
        )
    }

    private fun generateGametes(genotype: Genotype): List<String> {
        // This would be more complex for dihybrid crosses
        return listOf(genotype.alleles[0].toString(), genotype.alleles[1].toString())
    }

    private fun combineGametes(gamete1: String, gamete2: String): Genotype {
        return Genotype(String(charArrayOf(gamete1[0], gamete2[0]).sortedArray()))
    }

    private fun calculatePhenotypeRatios(
        offspring: List<com.animalgenetics.domain.model.Genotype>,
        traits: List<com.animalgenetics.domain.model.Trait>
    ): Map<String, Float> {
        val trait = (traits.first() as Trait)
        val dominantPhenotypeCount = offspring.count {
            (it as Genotype).alleles.contains(trait.dominantAllele)
        }
        val recessivePhenotypeCount = offspring.size - dominantPhenotypeCount

        return mapOf(
            "Dominant" to dominantPhenotypeCount.toFloat() / offspring.size,
            "Recessive" to recessivePhenotypeCount.toFloat() / offspring.size
        )
    }

    private fun calculateGenotypeRatios(offspring: List<com.animalgenetics.domain.model.Genotype>): Map<String, Float> {
        return offspring.groupingBy { (it as Genotype).alleles }
            .eachCount()
            .mapValues { it.value.toFloat() / offspring.size }
    }
}
