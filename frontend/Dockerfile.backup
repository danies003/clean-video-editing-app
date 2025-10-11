FROM node:18

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --silent

# Copy source code
COPY . .

# Set environment variables for build
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Make build script executable and run it
RUN chmod +x build-docker.sh && ./build-docker.sh

# Verify build output exists
RUN ls -la .next/

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"] 