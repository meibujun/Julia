package com.example.sfms.security.jwt;

import com.example.sfms.security.UserDetailsImpl;
import io.jsonwebtoken.*;
import io.jsonwebtoken.security.Keys;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.core.Authentication;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.util.Date;

@Component
public class JwtUtils {
    private static final Logger logger = LoggerFactory.getLogger(JwtUtils.class);

    @Value("\${sfms.app.jwtSecret:DefaultSecretKeyWhichShouldBeLongAndSecureAndAtLeast256Bits}") // Provide a default or ensure it's in application.properties
    private String jwtSecretString;

    @Value("\${sfms.app.jwtExpirationMs:86400000}") // Default to 1 day
    private int jwtExpirationMs;

    private SecretKey key;

    @jakarta.annotation.PostConstruct
    public void init() {
        if (jwtSecretString == null || jwtSecretString.length() < 32 || "DefaultSecretKeyWhichShouldBeLongAndSecureAndAtLeast256Bits".equals(jwtSecretString)) {
             logger.warn("JWT secret is not configured, too short, or still the default. Using a default insecure key for now. PLEASE CONFIGURE a strong sfms.app.jwtSecret in application.properties");
             // Using a generated key for safety if not configured, but still log warning.
             this.key = Keys.secretKeyFor(SignatureAlgorithm.HS512);
        } else {
             byte[] keyBytes = jwtSecretString.getBytes();
             this.key = Keys.hmacShaKeyFor(keyBytes);
        }
    }


    public String generateJwtToken(Authentication authentication) {
        UserDetailsImpl userPrincipal = (UserDetailsImpl) authentication.getPrincipal();

        return Jwts.builder()
                .setSubject((userPrincipal.getUsername()))
                .setIssuedAt(new Date())
                .setExpiration(new Date((new Date()).getTime() + jwtExpirationMs))
                .signWith(key, SignatureAlgorithm.HS512)
                .compact();
    }

    public String generateTokenFromUsername(String username) {
        return Jwts.builder().setSubject(username).setIssuedAt(new Date())
            .setExpiration(new Date((new Date()).getTime() + jwtExpirationMs)).signWith(key, SignatureAlgorithm.HS512)
            .compact();
    }


    public String getUserNameFromJwtToken(String token) {
        return Jwts.parserBuilder().setSigningKey(key).build().parseClaimsJws(token).getBody().getSubject();
    }

    public boolean validateJwtToken(String authToken) {
        try {
            Jwts.parserBuilder().setSigningKey(key).build().parseClaimsJws(authToken);
            return true;
        } catch (MalformedJwtException e) {
            logger.error("Invalid JWT token: {}", e.getMessage());
        } catch (ExpiredJwtException e) {
            logger.error("JWT token is expired: {}", e.getMessage());
        } catch (UnsupportedJwtException e) {
            logger.error("JWT token is unsupported: {}", e.getMessage());
        } catch (IllegalArgumentException e) {
            logger.error("JWT claims string is empty: {}", e.getMessage());
        } // Add SignatureException for HS512 if needed, though parserBuilder should handle key related issues.
          catch (io.jsonwebtoken.security.SignatureException e) {
            logger.error("Invalid JWT signature: {}", e.getMessage());
        }


        return false;
    }
}
