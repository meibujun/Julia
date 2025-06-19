package com.example.sfms.security;

import com.example.sfms.security.jwt.JwtUtils;
import io.jsonwebtoken.ExpiredJwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.MalformedJwtException;
import io.jsonwebtoken.SignatureAlgorithm;
import io.jsonwebtoken.security.Keys;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;

import org.springframework.security.core.Authentication;
import org.springframework.test.util.ReflectionTestUtils; // For setting @Value fields

import java.util.Date;
import java.util.Set; // Added for UserDetailsImpl constructor
import javax.crypto.SecretKey;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.when;
import org.mockito.junit.jupiter.MockitoExtension;


@ExtendWith(MockitoExtension.class)
class JwtUtilsTest {

    // We test JwtUtils directly, not via Spring context for @Value, so we set them manually.
    private JwtUtils jwtUtils;

    @Mock
    private Authentication authentication;

    private SecretKey testKey;

    @BeforeEach
    void setUp() {
        jwtUtils = new JwtUtils();
        // Use a fixed, known key for testing
        String secret = "TestSecretKeyWhichIsDefinitelyLongEnoughAndSecureForTesting12345";
        testKey = Keys.hmacShaKeyFor(secret.getBytes());
        ReflectionTestUtils.setField(jwtUtils, "jwtSecretString", secret);
        ReflectionTestUtils.setField(jwtUtils, "jwtExpirationMs", 3600000); // 1 hour
        jwtUtils.init(); // Manually call init to set up the key from jwtSecretString
    }

    @Test
    void generateJwtToken_success() {
        UserDetailsImpl userDetails = new UserDetailsImpl(1L, "testuser", "test@example.com", "pass", true, Set.of());
        when(authentication.getPrincipal()).thenReturn(userDetails);

        String token = jwtUtils.generateJwtToken(authentication);
        assertNotNull(token);
        assertEquals("testuser", jwtUtils.getUserNameFromJwtToken(token));
    }

    @Test
    void validateJwtToken_validToken_returnsTrue() {
        UserDetailsImpl userDetails = new UserDetailsImpl(1L, "testuser", "test@example.com", "pass", true, Set.of());
        when(authentication.getPrincipal()).thenReturn(userDetails);
        String token = jwtUtils.generateJwtToken(authentication);

        assertTrue(jwtUtils.validateJwtToken(token));
    }

    @Test
    void validateJwtToken_invalidToken_returnsFalse() {
        assertFalse(jwtUtils.validateJwtToken("invalid.token.string"));
    }

    @Test
    void validateJwtToken_malformedToken_logsErrorAndReturnsFalse() {
        // A token that is not even in the correct format
        String malformedToken = "thisIsNotAJWT";
        assertFalse(jwtUtils.validateJwtToken(malformedToken));
        // Add log capture verification if needed
    }

    @Test
    void validateJwtToken_expiredToken_returnsFalse() {
        // Generate a token that expired in the past
        String expiredToken = Jwts.builder()
                .setSubject("expiredUser")
                .setIssuedAt(new Date(System.currentTimeMillis() - 2000000)) // 2000 seconds ago
                .setExpiration(new Date(System.currentTimeMillis() - 1000000)) // 1000 seconds ago
                .signWith(testKey, SignatureAlgorithm.HS512)
                .compact();

        // parsing will throw ExpiredJwtException, validateJwtToken should catch it and return false.
        assertFalse(jwtUtils.validateJwtToken(expiredToken));
    }

    @Test
    void getUserNameFromJwtToken_success() {
        UserDetailsImpl userDetails = new UserDetailsImpl(1L, "testuser", "test@example.com", "pass", true, Set.of());
        when(authentication.getPrincipal()).thenReturn(userDetails);
        String token = jwtUtils.generateJwtToken(authentication);
        assertEquals("testuser", jwtUtils.getUserNameFromJwtToken(token));
    }

    @Test
    void getUserNameFromJwtToken_expiredToken_throwsExpiredJwtException() {
        // Generate a token that expired in the past
        String expiredToken = Jwts.builder()
                .setSubject("expiredUser")
                .setIssuedAt(new Date(System.currentTimeMillis() - 2000000))
                .setExpiration(new Date(System.currentTimeMillis() - 1000000))
                .signWith(testKey, SignatureAlgorithm.HS512)
                .compact();

        // Directly calling getUserNameFromJwtToken should throw the exception if token is expired.
        assertThrows(ExpiredJwtException.class, () -> jwtUtils.getUserNameFromJwtToken(expiredToken));
    }
}
