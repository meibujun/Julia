package com.animalgenetics.domain.model

interface CaseStudy {
    val id: String
    val title: String
    val description: String
}

data class InteractiveCaseSession(
    val id: String,
    val caseStudy: CaseStudy,
    val userId: String,
    val groupId: String?,
    var currentPhase: CasePhase,
    val decisions: MutableList<CaseDecision>,
    val evidence: MutableList<Any>
)

enum class CasePhase {
    INTRODUCTION,
    DATA_GATHERING,
    ANALYSIS,
    CONCLUSION,
    FEEDBACK
}

data class CaseDecision(
    val phase: CasePhase,
    val decision: String, // e.g., "Order Test X"
    val rationale: String
)

data class DecisionResult(
    val consequences: String,
    val feedback: String,
    val newPhase: CasePhase,
    val unlockedInformation: List<Any>
)

// Placeholders for the detailed genetics case study
data class GeneticsScenario(val background: String)
data class PatientGeneticData(val sequence: String)
data class GeneticTest(val name: String, val cost: Int)
data class EthicalIssue(val description: String)
data class AnalysisResult(
    val testResults: List<Any>,
    val diagnosis: String,
    val recommendations: List<String>,
    val confidenceLevel: Float
)
