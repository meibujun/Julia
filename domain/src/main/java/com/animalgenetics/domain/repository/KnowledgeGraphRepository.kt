package com.animalgenetics.domain.repository

import com.animalgenetics.domain.model.GeneticsConcept

interface KnowledgeGraphRepository {
    suspend fun getRelatedConcepts(topic: GeneticsConcept, maxDepth: Int): List<GeneticsConcept>
    suspend fun getConceptById(conceptId: String): GeneticsConcept?
    suspend fun syncConcepts(courseId: String): List<GeneticsConcept>
}
