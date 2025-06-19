package com.example.sfms.controller;

import com.example.sfms.dto.RoleRequestDto;
import com.example.sfms.dto.RoleResponseDto;
import com.example.sfms.service.RoleService;
import com.example.sfms.entity.Role;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
// Import Spring Security annotations for authorization later e.g. @PreAuthorize("hasRole('ADMIN')")

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/roles")
// @PreAuthorize("hasRole('ADMIN')") // Secure all role endpoints for ADMIN only
public class RoleController {

    private final RoleService roleService;

    @Autowired
    public RoleController(RoleService roleService) {
        this.roleService = roleService;
    }

    private RoleResponseDto convertToDto(Role role) {
        return new RoleResponseDto(role.getId(), role.getName());
    }

    @PostMapping
    public ResponseEntity<?> createRole(@Valid @RequestBody RoleRequestDto roleRequestDto) {
        try {
            Role role = new Role(roleRequestDto.getName());
            Role createdRole = roleService.createRole(role);
            return ResponseEntity.status(HttpStatus.CREATED).body(convertToDto(createdRole));
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage());
        }
    }

    @GetMapping("/{id}")
    public ResponseEntity<?> getRoleById(@PathVariable Long id) {
        return roleService.findById(id)
                .map(role -> ResponseEntity.ok(convertToDto(role)))
                .orElse(ResponseEntity.status(HttpStatus.NOT_FOUND).body("Role not found with id: " + id));
    }

    @GetMapping
    public ResponseEntity<List<RoleResponseDto>> getAllRoles() {
        List<RoleResponseDto> roles = roleService.findAllRoles().stream()
                .map(this::convertToDto)
                .collect(Collectors.toList());
        return ResponseEntity.ok(roles);
    }

    @PutMapping("/{id}")
    public ResponseEntity<?> updateRole(@PathVariable Long id, @Valid @RequestBody RoleRequestDto roleRequestDto) {
        try {
            Role roleDetails = new Role(roleRequestDto.getName());
            // The ID from path should be used, not from roleDetails if it had an ID field.
            Role updatedRole = roleService.updateRole(id, roleDetails);
            return ResponseEntity.ok(convertToDto(updatedRole));
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage());
        }
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<?> deleteRole(@PathVariable Long id) {
        try {
            roleService.deleteRole(id);
            return ResponseEntity.ok("Role deleted successfully with id: " + id);
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(e.getMessage());
        }
    }
}
