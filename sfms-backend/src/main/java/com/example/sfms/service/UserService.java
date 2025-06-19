package com.example.sfms.service;

import com.example.sfms.entity.User;
import com.example.sfms.dto.UserRegistrationRequestDto; // Assuming DTO will be created
import java.util.List;
import java.util.Optional;
import java.util.Set; // Added import for Set

public interface UserService {
    User registerUser(UserRegistrationRequestDto registrationDto); // DTO will be defined later
    Optional<User> findById(Long id);
    Optional<User> findByUsername(String username);
    Optional<User> findByEmail(String email);
    List<User> findAllUsers();
    User updateUser(Long id, User userDetails); // Consider a DTO for update as well
    void deleteUser(Long id);
    User assignRolesToUser(Long userId, Set<String> roleNames); // Using role names for simplicity
    // Add more methods as needed, e.g., changePassword, activateUser etc.
}
