package com.example.sfms.entity;

import jakarta.persistence.*;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.PastOrPresent;
import jakarta.validation.constraints.Size;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "sheep", uniqueConstraints = {
    @UniqueConstraint(columnNames = "ear_tag_number")
})
public class Sheep {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @NotBlank(message = "Ear tag number cannot be blank")
    @Size(max = 50, message = "Ear tag number must be less than 50 characters")
    @Column(name = "ear_tag_number", nullable = false, unique = true, length = 50)
    private String earTagNumber;

    @Size(max = 100, message = "Breed must be less than 100 characters")
    @Column(length = 100)
    private String breed;

    @NotNull(message = "Sex cannot be null")
    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 10)
    private Sex sex;

    @PastOrPresent(message = "Birth date must be in the past or present")
    @Column(name = "birth_date")
    private LocalDate birthDate;

    @Column(name = "dam_id")
    private Long damId; // Foreign key to another Sheep (mother)

    @Column(name = "sire_id")
    private Long sireId; // Foreign key to another Sheep (father)

    @Size(max = 50, message = "Health status must be less than 50 characters")
    @Column(name = "health_status", length = 50)
    private String healthStatus; // Could be an Enum: HEALTHY, SICK, QUARANTINED etc.

    @Lob // For longer text
    private String notes;

    @NotNull(message = "Entry date cannot be null")
    @PastOrPresent(message = "Entry date must be in the past or present")
    @Column(name = "entry_date", nullable = false)
    private LocalDate entryDate;

    @PastOrPresent(message = "Exit date must be in the past or present")
    @Column(name = "exit_date")
    private LocalDate exitDate;

    @Size(max = 100, message = "Exit reason must be less than 100 characters")
    @Column(name = "exit_reason", length = 100)
    private String exitReason;

    // @Column(name = "farm_id") // Uncomment if multi-farm support is needed
    // private Long farmId;

    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    // Constructors
    public Sheep() {
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getEarTagNumber() { return earTagNumber; }
    public void setEarTagNumber(String earTagNumber) { this.earTagNumber = earTagNumber; }

    public String getBreed() { return breed; }
    public void setBreed(String breed) { this.breed = breed; }

    public Sex getSex() { return sex; }
    public void setSex(Sex sex) { this.sex = sex; }

    public LocalDate getBirthDate() { return birthDate; }
    public void setBirthDate(LocalDate birthDate) { this.birthDate = birthDate; }

    public Long getDamId() { return damId; }
    public void setDamId(Long damId) { this.damId = damId; }

    public Long getSireId() { return sireId; }
    public void setSireId(Long sireId) { this.sireId = sireId; }

    public String getHealthStatus() { return healthStatus; }
    public void setHealthStatus(String healthStatus) { this.healthStatus = healthStatus; }

    public String getNotes() { return notes; }
    public void setNotes(String notes) { this.notes = notes; }

    public LocalDate getEntryDate() { return entryDate; }
    public void setEntryDate(LocalDate entryDate) { this.entryDate = entryDate; }

    public LocalDate getExitDate() { return exitDate; }
    public void setExitDate(LocalDate exitDate) { this.exitDate = exitDate; }

    public String getExitReason() { return exitReason; }
    public void setExitReason(String exitReason) { this.exitReason = exitReason; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    // Lifecycle Callbacks
    @PrePersist
    protected void onCreate() {
        LocalDateTime now = LocalDateTime.now();
        createdAt = now;
        updatedAt = now;
        if (this.entryDate == null) { // Default entry date to today if not set
            this.entryDate = LocalDate.now();
        }
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    // toString, equals, hashCode (optional but good practice)
    @Override
    public String toString() {
        return "Sheep{" +
                "id=" + id +
                ", earTagNumber='" + earTagNumber + '\'' +
                ", sex=" + sex +
                ", birthDate=" + birthDate +
                ", breed='" + breed + '\'' +
                '}';
    }

    // Consider implementing equals and hashCode based on earTagNumber or id
}
