# Use a minimal Rust image to install mdBook
FROM rust:alpine AS builder

# Install build dependencies
RUN apk add --no-cache musl-dev openssl-dev

# Install mdBook
RUN cargo install mdbook

# Create a minimal runtime image
FROM alpine:latest

# Install runtime dependencies
RUN apk add --no-cache libgcc

# Copy mdBook binary from builder
COPY --from=builder /usr/local/cargo/bin/mdbook /usr/local/bin/mdbook

# Set workdir to /book (optional)
WORKDIR /book

# Expose port for mdbook serve (default: 3000)
EXPOSE 3000

# Default command (can be overridden)
CMD ["mdbook", "serve", "-n", "0.0.0.0", "-p", "6969"]

# docker build -t mdbook .
# docker run -p 6969:6969 -v $(pwd):/book mdbook
