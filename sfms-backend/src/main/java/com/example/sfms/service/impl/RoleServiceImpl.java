package com.example.sfms.service.impl;

import com.example.sfms.entity.Role;
import com.example.sfms.repository.RoleRepository;
import com.example.sfms.service.RoleService;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.List;
import java.util.Optional;

@Service
public class RoleServiceImpl implements RoleService {

    private final RoleRepository roleRepository;

    @Autowired
    public RoleServiceImpl(RoleRepository roleRepository) {
        this.roleRepository = roleRepository;
    }

    @Override
    @Transactional
    public Role createRole(Role role) { // Consider RoleCreationDto
        if (roleRepository.existsByName(role.getName())) {
            throw new RuntimeException("Error: Role name is already taken!"); // Replace with custom exception
        }
        return roleRepository.save(role);
    }

    @Override
    public Optional<Role> findById(Long id) {
        return roleRepository.findById(id);
    }

    @Override
    public Optional<Role> findByName(String name) {
        return roleRepository.findByName(name);
    }

    @Override
    public List<Role> findAllRoles() {
        return roleRepository.findAll();
    }

    @Override
    @Transactional
    public Role updateRole(Long id, Role roleDetails) { // Consider RoleUpdateDto
        Role role = roleRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Error: Role not found with id " + id));

        // Check if new name is taken by another role
        if (!role.getName().equals(roleDetails.getName()) && roleRepository.existsByName(roleDetails.getName())) {
            throw new RuntimeException("Error: New role name is already taken!");
        }
        role.setName(roleDetails.getName());
        // Add other updatable fields if any
        return roleRepository.save(role);
    }

    @Override
    @Transactional
    public void deleteRole(Long id) {
        Role role = roleRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Error: Role not found with id " + id));

        // Optional: Check if role is assigned to any users before deletion
        // if (!role.getUsers().isEmpty()) {
        //    throw new RuntimeException("Error: Cannot delete role. It is assigned to active users.");
        // }
        roleRepository.deleteById(id);
    }
}
