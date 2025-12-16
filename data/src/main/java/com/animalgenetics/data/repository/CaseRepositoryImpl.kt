package com.animalgenetics.data.repository

import com.animalgenetics.domain.model.CaseStudy
import com.animalgenetics.domain.repository.CaseRepository
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class CaseRepositoryImpl @Inject constructor() : CaseRepository {
    override suspend fun getCase(caseId: String): CaseStudy {
        // Mock implementation
        return object : CaseStudy {
            override val id: String = caseId
            override val title: String = "Mock Case"
            override val description: String = "This is a mock case study."
        }
    }
}
