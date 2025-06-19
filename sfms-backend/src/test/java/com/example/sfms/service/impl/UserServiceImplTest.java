package com.example.sfms.service.impl;

import com.example.sfms.dto.UserRegistrationRequestDto;
import com.example.sfms.entity.Role;
import com.example.sfms.entity.User;
import com.example.sfms.repository.RoleRepository;
import com.example.sfms.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.security.crypto.password.PasswordEncoder;

import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class UserServiceImplTest {

    @Mock
    private UserRepository userRepository;

    @Mock
    private RoleRepository roleRepository;

    @Mock
    private PasswordEncoder passwordEncoder;

    @InjectMocks
    private UserServiceImpl userService;

    private UserRegistrationRequestDto registrationDto;
    private User user;
    private Role userRole;

    @BeforeEach
    void setUp() {
        registrationDto = new UserRegistrationRequestDto();
        registrationDto.setUsername("testuser");
        registrationDto.setEmail("test@example.com");
        registrationDto.setPassword("password123");
        registrationDto.setRoles(Set.of("ROLE_USER"));

        user = new User();
        user.setId(1L);
        user.setUsername("testuser");
        user.setEmail("test@example.com");
        user.setPassword("encodedPassword");

        userRole = new Role("ROLE_USER");
        userRole.setId(1L);
    }

    @Test
    void registerUser_success() {
        when(userRepository.existsByUsername(anyString())).thenReturn(false);
        when(userRepository.existsByEmail(anyString())).thenReturn(false);
        when(passwordEncoder.encode(anyString())).thenReturn("encodedPassword");
        when(roleRepository.findByName("ROLE_USER")).thenReturn(Optional.of(userRole));
        when(userRepository.save(any(User.class))).thenAnswer(invocation -> {
            User savedUser = invocation.getArgument(0);
            savedUser.setId(1L); // Simulate saving and getting an ID
            return savedUser;
        });

        User registeredUser = userService.registerUser(registrationDto);

        assertNotNull(registeredUser);
        assertEquals("testuser", registeredUser.getUsername());
        assertEquals("encodedPassword", registeredUser.getPassword());
        assertTrue(registeredUser.getRoles().contains(userRole));
        verify(userRepository, times(1)).save(any(User.class));
    }

    @Test
    void registerUser_usernameExists_throwsException() {
        when(userRepository.existsByUsername("testuser")).thenReturn(true);

        Exception exception = assertThrows(RuntimeException.class, () -> {
            userService.registerUser(registrationDto);
        });
        assertEquals("Error: Username is already taken!", exception.getMessage());
        verify(userRepository, never()).save(any(User.class));
    }

    @Test
    void registerUser_emailExists_throwsException() {
        when(userRepository.existsByUsername(anyString())).thenReturn(false);
        when(userRepository.existsByEmail("test@example.com")).thenReturn(true);

        Exception exception = assertThrows(RuntimeException.class, () -> {
            userService.registerUser(registrationDto);
        });
        assertEquals("Error: Email is already in use!", exception.getMessage());
        verify(userRepository, never()).save(any(User.class));
    }

    @Test
    void registerUser_defaultRoleAssigned_whenNoRolesProvided() {
        registrationDto.setRoles(null); // No roles provided
        when(userRepository.existsByUsername(anyString())).thenReturn(false);
        when(userRepository.existsByEmail(anyString())).thenReturn(false);
        when(passwordEncoder.encode(anyString())).thenReturn("encodedPassword");
        when(roleRepository.findByName("ROLE_USER")).thenReturn(Optional.of(userRole)); // Default role
        when(userRepository.save(any(User.class))).thenReturn(user);

        User registeredUser = userService.registerUser(registrationDto);

        assertNotNull(registeredUser);
        assertTrue(registeredUser.getRoles().stream().anyMatch(role -> role.getName().equals("ROLE_USER")));
        verify(userRepository, times(1)).save(any(User.class));
    }


    @Test
    void findByUsername_userExists_returnsUser() {
        when(userRepository.findByUsername("testuser")).thenReturn(Optional.of(user));
        Optional<User> foundUser = userService.findByUsername("testuser");
        assertTrue(foundUser.isPresent());
        assertEquals("testuser", foundUser.get().getUsername());
    }

    @Test
    void findByUsername_userNotExists_returnsEmpty() {
        when(userRepository.findByUsername("nonexistent")).thenReturn(Optional.empty());
        Optional<User> foundUser = userService.findByUsername("nonexistent");
        assertFalse(foundUser.isPresent());
    }
}
