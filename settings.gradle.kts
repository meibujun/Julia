pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "AnimalGeneticsApp"
include(":app")
include(":core")
include(":data")
include(":domain")

// Feature Modules
include(":feature:auth")
include(":feature:knowledge-graph")
include(":feature:ai-tutor")
include(":feature:virtual-lab")
include(":feature:assessment")
include(":feature:gamification")
include(":feature:case-studies")

// Core Sub-Modules (as an example of further modularization if needed)
// include(":core:common")
// include(":core:database")
// include(":core:network")
// include(":core:ui")
