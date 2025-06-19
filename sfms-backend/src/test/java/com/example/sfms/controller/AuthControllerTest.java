package com.example.sfms.controller;

import com.example.sfms.dto.LoginRequestDto;
import com.example.sfms.dto.UserRegistrationRequestDto;
import com.example.sfms.entity.Role;
import com.example.sfms.entity.User;
import com.example.sfms.repository.RoleRepository;
import com.example.sfms.repository.UserRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.transaction.annotation.Transactional;

import java.util.Set;

import static org.hamcrest.Matchers.is;
import static org.hamcrest.Matchers.notNullValue;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD) // Reset context for each test
@Transactional // Rollback transactions after each test
class AuthControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private RoleRepository roleRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    private Role userRole;

    @BeforeEach
    void setUp() {
        // Ensure roles exist
        userRole = roleRepository.findByName("ROLE_USER").orElseGet(() -> roleRepository.save(new Role("ROLE_USER")));
        roleRepository.findByName("ROLE_ADMIN").orElseGet(() -> roleRepository.save(new Role("ROLE_ADMIN")));
    }

    @Test
    void registerUser_success() throws Exception {
        UserRegistrationRequestDto registrationDto = new UserRegistrationRequestDto();
        registrationDto.setUsername("newuser");
        registrationDto.setEmail("newuser@example.com");
        registrationDto.setPassword("password123");
        registrationDto.setRoles(Set.of("ROLE_USER"));

        mockMvc.perform(post("/api/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(registrationDto)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.username", is("newuser")))
                .andExpect(jsonPath("$.email", is("newuser@example.com")))
                .andExpect(jsonPath("$.roles[0]", is("ROLE_USER"))); // Assuming roles set has defined order or single element

        assertTrue(userRepository.existsByUsername("newuser"));
    }

    @Test
    void registerUser_usernameTaken() throws Exception {
        // Arrange: existing user
        User existingUser = new User("existinguser", passwordEncoder.encode("password"), "existing@example.com");
        existingUser.getRoles().add(userRole);
        userRepository.save(existingUser);

        UserRegistrationRequestDto registrationDto = new UserRegistrationRequestDto();
        registrationDto.setUsername("existinguser"); // Username that already exists
        registrationDto.setEmail("newemail@example.com");
        registrationDto.setPassword("password123");

        mockMvc.perform(post("/api/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(registrationDto)))
                .andExpect(status().isBadRequest())
                .andExpect(content().string("Error: Username is already taken!"));
    }

    @Test
    void registerUser_invalidData_validationFails() throws Exception {
        UserRegistrationRequestDto registrationDto = new UserRegistrationRequestDto();
        registrationDto.setUsername("u"); // Too short
        registrationDto.setEmail("not-an-email");
        registrationDto.setPassword("short"); // Matches length, but other fields fail

        mockMvc.perform(post("/api/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(registrationDto)))
                .andExpect(status().isBadRequest()); // Spring Boot validation should return 400
                // More specific checks for validation messages can be added
    }

    @Test
    void loginUser_success() throws Exception {
        // Arrange: Create a user first
        User testUser = new User("loginuser", passwordEncoder.encode("password123"), "login@example.com");
        testUser.getRoles().add(userRole);
        testUser.setActive(true);
        userRepository.save(testUser);

        LoginRequestDto loginDto = new LoginRequestDto();
        loginDto.setUsername("loginuser");
        loginDto.setPassword("password123");

        mockMvc.perform(post("/api/auth/login")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(loginDto)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.token", notNullValue()))
                .andExpect(jsonPath("$.username", is("loginuser")))
                .andExpect(jsonPath("$.roles[0]", is("ROLE_USER")));
    }

    @Test
    void loginUser_invalidCredentials_unauthorized() throws Exception {
        LoginRequestDto loginDto = new LoginRequestDto();
        loginDto.setUsername("nonexistentuser");
        loginDto.setPassword("wrongpassword");

        mockMvc.perform(post("/api/auth/login")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(loginDto)))
                .andExpect(status().isUnauthorized()); // Spring Security's default for bad credentials
    }
}
