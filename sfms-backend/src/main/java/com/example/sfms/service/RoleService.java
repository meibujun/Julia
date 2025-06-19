package com.example.sfms.service;

import com.example.sfms.entity.Role;
import java.util.List;
import java.util.Optional;

public interface RoleService {
    Role createRole(Role role); // Consider a RoleDto
    Optional<Role> findById(Long id);
    Optional<Role> findByName(String name);
    List<Role> findAllRoles();
    Role updateRole(Long id, Role roleDetails); // Consider a RoleDto
    void deleteRole(Long id);
    // Add more methods as needed
}
