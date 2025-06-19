package com.example.sfms.service.impl;

import com.example.sfms.entity.Role;
import com.example.sfms.entity.User;
import com.example.sfms.dto.UserRegistrationRequestDto;
import com.example.sfms.repository.RoleRepository;
import com.example.sfms.repository.UserRepository;
import com.example.sfms.service.UserService;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

@Service
public class UserServiceImpl implements UserService {

    private final UserRepository userRepository;
    private final RoleRepository roleRepository;
    private final PasswordEncoder passwordEncoder;

    @Autowired
    public UserServiceImpl(UserRepository userRepository, RoleRepository roleRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.roleRepository = roleRepository;
        this.passwordEncoder = passwordEncoder;
    }

    @Override
    @Transactional
    public User registerUser(UserRegistrationRequestDto registrationDto) {
        if (userRepository.existsByUsername(registrationDto.getUsername())) {
            throw new RuntimeException("Error: Username is already taken!"); // Replace with custom exception
        }

        if (userRepository.existsByEmail(registrationDto.getEmail())) {
            throw new RuntimeException("Error: Email is already in use!"); // Replace with custom exception
        }

        User user = new User();
        user.setUsername(registrationDto.getUsername());
        user.setPassword(passwordEncoder.encode(registrationDto.getPassword()));
        user.setEmail(registrationDto.getEmail());
        user.setActive(true); // Default to active

        Set<String> strRoles = registrationDto.getRoles();
        Set<Role> roles = new HashSet<>();

        if (strRoles == null || strRoles.isEmpty()) {
            Role userRole = roleRepository.findByName("ROLE_USER") // Assuming "ROLE_USER" is a default role
                    .orElseThrow(() -> new RuntimeException("Error: Role USER is not found."));
            roles.add(userRole);
        } else {
            strRoles.forEach(roleName -> {
                Role role = roleRepository.findByName(roleName)
                        .orElseThrow(() -> new RuntimeException("Error: Role " + roleName + " is not found."));
                roles.add(role);
            });
        }
        user.setRoles(roles);
        return userRepository.save(user);
    }

    @Override
    public Optional<User> findById(Long id) {
        return userRepository.findById(id);
    }

    @Override
    public Optional<User> findByUsername(String username) {
        return userRepository.findByUsername(username);
    }

    @Override
    public Optional<User> findByEmail(String email) {
        return userRepository.findByEmail(email);
    }

    @Override
    public List<User> findAllUsers() {
        return userRepository.findAll();
    }

    @Override
    @Transactional
    public User updateUser(Long id, User userDetails) { // Consider UserUpdateDto
        User user = userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Error: User not found with id " + id));

        user.setEmail(userDetails.getEmail()); // Add more updatable fields
        user.setActive(userDetails.isActive());
        // Password change should be a separate, secure method
        // if (userDetails.getPassword() != null && !userDetails.getPassword().isEmpty()) {
        //    user.setPassword(passwordEncoder.encode(userDetails.getPassword()));
        // }
        if (userDetails.getRoles() != null && !userDetails.getRoles().isEmpty()) {
             Set<Role> updatedRoles = userDetails.getRoles().stream()
                .map(role -> roleRepository.findByName(role.getName())
                                .orElseThrow(() -> new RuntimeException("Error: Role " + role.getName() + " not found.")))
                .collect(Collectors.toSet());
            user.setRoles(updatedRoles);
        }

        return userRepository.save(user);
    }

    @Override
    @Transactional
    public void deleteUser(Long id) {
        if (!userRepository.existsById(id)) {
            throw new RuntimeException("Error: User not found with id " + id);
        }
        userRepository.deleteById(id);
    }

    @Override
    @Transactional
    public User assignRolesToUser(Long userId, Set<String> roleNames) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("Error: User not found with id " + userId));

        Set<Role> roles = roleNames.stream()
                .map(roleName -> roleRepository.findByName(roleName)
                        .orElseThrow(() -> new RuntimeException("Error: Role " + roleName + " is not found.")))
                .collect(Collectors.toSet());

        user.setRoles(roles);
        return userRepository.save(user);
    }
}
