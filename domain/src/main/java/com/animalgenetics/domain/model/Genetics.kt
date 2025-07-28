package com.animalgenetics.domain.model

data class GeneticsConcept(
    val id: String,
    val name: String,
    val type: ConceptType,
    var userMastery: Float, // Value between 0.0 and 1.0
    val difficulty: Int, // e.g., 1 to 5
    val prerequisites: List<String> // List of concept IDs
)

enum class ConceptType {
    DEFINITION,
    LAW,
    MECHANISM,
    CASE_STUDY
}

data class ConceptRelationship(
    val fromId: String,
    val toId: String,
    val type: RelationshipType,
    val strength: Float
)

enum class RelationshipType {
    PREREQUISITE,
    RELATED,
    EXAMPLE_OF
}

data class GraphData(
    val nodes: List<GraphNode>,
    val edges: List<GraphEdge>
)

data class GraphNode(
    val id: String,
    val label: String,
    val type: String,
    val mastery: Float,
    val properties: Map<String, Any>
)

data class GraphEdge(
    val source: String,
    val target: String,
    val type: String,
    val weight: Float
)
