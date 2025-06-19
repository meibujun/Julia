package com.example.sfms.controller;

import com.example.sfms.dto.UserResponseDto;
import com.example.sfms.dto.UserUpdateRequestDto;
import com.example.sfms.service.UserService;
import com.example.sfms.entity.User;
import com.example.sfms.entity.Role;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
// Import Spring Security annotations for authorization later e.g. @PreAuthorize
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/users")
public class UserController {

    private final UserService userService;

    @Autowired
    public UserController(UserService userService) {
        this.userService = userService;
    }

    // Helper method to convert User entity to UserResponseDto
    private UserResponseDto convertToDto(User user) {
        return new UserResponseDto(
                user.getId(),
                user.getUsername(),
                user.getEmail(),
                user.isActive(),
                user.getRoles().stream().map(Role::getName).collect(Collectors.toSet()),
                user.getCreatedAt(),
                user.getUpdatedAt()
        );
    }

    @GetMapping("/{id}")
    // @PreAuthorize("hasRole('ADMIN') or @userService.isOwner(authentication, #id)") // Example for future
    public ResponseEntity<?> getUserById(@PathVariable Long id) {
        return userService.findById(id)
                .map(user -> ResponseEntity.ok(convertToDto(user)))
                .orElse(ResponseEntity.status(HttpStatus.NOT_FOUND).body("User not found with id: " + id));
    }

    @GetMapping
    // @PreAuthorize("hasRole('ADMIN')") // Example for future
    public ResponseEntity<List<UserResponseDto>> getAllUsers() {
        List<UserResponseDto> users = userService.findAllUsers().stream()
                .map(this::convertToDto)
                .collect(Collectors.toList());
        return ResponseEntity.ok(users);
    }

    // GET /api/users/me - to be implemented with Spring Security to get current user

    @PutMapping("/{id}")
    // @PreAuthorize("hasRole('ADMIN') or @userService.isOwner(authentication, #id)")
    public ResponseEntity<?> updateUser(@PathVariable Long id, @Valid @RequestBody UserUpdateRequestDto userUpdateDto) {
        try {
            // This is a simplified update. Service layer should handle partial updates.
            User userDetails = new User(); // Temporary User object for passing data
            if (userUpdateDto.getEmail() != null) userDetails.setEmail(userUpdateDto.getEmail());
            if (userUpdateDto.getIsActive() != null) userDetails.setActive(userUpdateDto.getIsActive());
            if (userUpdateDto.getRoles() != null) {
                // This part of mapping (string roles to Role entities) should ideally be in the service
                // For now, this DTO only passes role names. The service's updateUser needs to handle resolving them.
                // Or, the service's updateUser method should accept UserUpdateRequestDto directly.
                // For simplicity here, we assume service's updateUser can take a User object with partial updates.
            }


            User userToUpdate = userService.findById(id)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + id));

            // Apply updates from DTO
            if(userUpdateDto.getEmail() != null) userToUpdate.setEmail(userUpdateDto.getEmail());
            if(userUpdateDto.getIsActive() != null) userToUpdate.setActive(userUpdateDto.getIsActive());

            // Role update should be handled carefully, potentially by a dedicated method in service
            // or ensuring userService.updateUser can handle a Set<String> for roles from DTO.
            // For now, assuming a simplified updateUser or that roles are handled by assignRolesToUser
            User updatedUser = userService.updateUser(id, userToUpdate); // Pass the existing user with modified fields

            return ResponseEntity.ok(convertToDto(updatedUser));
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage());
        }
    }

    @PutMapping("/{id}/roles")
    // @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<?> assignRolesToUser(@PathVariable Long id, @RequestBody Set<String> roleNames) {
        try {
            User updatedUser = userService.assignRolesToUser(id, roleNames);
            return ResponseEntity.ok(convertToDto(updatedUser));
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage());
        }
    }


    @DeleteMapping("/{id}")
    // @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<?> deleteUser(@PathVariable Long id) {
        try {
            userService.deleteUser(id);
            return ResponseEntity.ok("User deleted successfully with id: " + id);
        } catch (RuntimeException e) {
            // Catch specific exceptions like UserNotFoundException if defined
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage());
        }
    }
}
