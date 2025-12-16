package com.animalgenetics.domain.usecase

import com.animalgenetics.domain.model.ConceptType
import com.animalgenetics.domain.model.GeneticsConcept
import com.animalgenetics.domain.repository.KnowledgeGraphRepository
import javax.inject.Inject

class GetGraphConceptsUseCase @Inject constructor(
    private val repository: KnowledgeGraphRepository
) {
    suspend fun execute(): Result<List<GeneticsConcept>> {
        // In the future, this would call repository.syncConcepts()
        // For the walking skeleton, we return hardcoded data.
        return Result.success(
            listOf(
                GeneticsConcept("1", "Mendelian Inheritance", ConceptType.LAW, 0.5f, 2, emptyList()),
                GeneticsConcept("2", "DNA Replication", ConceptType.MECHANISM, 0.2f, 3, listOf("1"))
            )
        )
    }
}
