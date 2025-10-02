#!/bin/bash

# Build script for Docker that handles @ imports
echo "🔧 Preparing build for Docker..."

# Create a temporary directory for the build
mkdir -p .build-temp

# Copy all files to temp directory
echo "📁 Copying files to temp directory..."
cp -r src .build-temp/
cp -r public .build-temp/
cp package*.json .build-temp/
cp next.config.ts .build-temp/
cp tsconfig.json .build-temp/
cp jsconfig.json .build-temp/

# Replace @ imports with relative imports using Node.js
echo "🔄 Converting @ imports to relative imports..."

# Use Node.js for reliable text replacement (without glob dependency)
cat > .build-temp/convert_imports.js << 'EOF'
const fs = require('fs');
const path = require('path');

function findFiles(dir, extensions) {
    let files = [];
    const items = fs.readdirSync(dir);
    
    for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        
        if (stat.isDirectory()) {
            files = files.concat(findFiles(fullPath, extensions));
        } else if (extensions.some(ext => item.endsWith(ext))) {
            files.push(fullPath);
        }
    }
    
    return files;
}

function convertImports(filePath) {
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Simple approach: replace @/ with appropriate relative paths based on file location
    if (filePath.startsWith('src/app/auth/callback/')) {
        // Files in src/app/auth/callback/ need to go up three levels
        content = content.replace(/from ['"]@\/lib\//g, 'from "../../../lib/');
        content = content.replace(/from ['"]@\/components\//g, 'from "../../../components/');
        content = content.replace(/from ['"]@\/app\//g, 'from "../../../app/');
        content = content.replace(/from ['"]@\//g, 'from "../../../');
        console.log(`Converted auth callback file: ${filePath}`);
    } else if (filePath.startsWith('src/app/test-segment-editor/') || filePath.startsWith('src/app/test-timeline/')) {
        // Files in subdirectories of src/app/ need to go up two levels
        content = content.replace(/from ['"]@\/lib\//g, 'from "../../lib/');
        content = content.replace(/from ['"]@\/components\//g, 'from "../../components/');
        content = content.replace(/from ['"]@\/app\//g, 'from "../../app/');
        content = content.replace(/from ['"]@\//g, 'from "../../');
        console.log(`Converted app subdir file: ${filePath}`);
    } else if (filePath.startsWith('src/app/')) {
        // Files in src/app/ need to go up one level to reach src/
        content = content.replace(/from ['"]@\/lib\//g, 'from "../lib/');
        content = content.replace(/from ['"]@\/components\//g, 'from "../components/');
        content = content.replace(/from ['"]@\/app\//g, 'from "../app/');
        content = content.replace(/from ['"]@\//g, 'from "../');
        console.log(`Converted app file: ${filePath}`);
    } else if (filePath.startsWith('src/components/ui/')) {
        // Files in src/components/ui/ need to go up two levels
        content = content.replace(/from ['"]@\/lib\//g, 'from "../../lib/');
        content = content.replace(/from ['"]@\/components\//g, 'from "../../components/');
        content = content.replace(/from ['"]@\/app\//g, 'from "../../app/');
        content = content.replace(/from ['"]@\//g, 'from "../../');
        console.log(`Converted UI component file: ${filePath}`);
    } else if (filePath.startsWith('src/components/')) {
        // Files in src/components/ need to go up one level
        content = content.replace(/from ['"]@\/lib\//g, 'from "../lib/');
        content = content.replace(/from ['"]@\/components\//g, 'from "../components/');
        content = content.replace(/from ['"]@\/app\//g, 'from "../app/');
        content = content.replace(/from ['"]@\//g, 'from "../');
        console.log(`Converted component file: ${filePath}`);
    } else if (filePath.startsWith('src/lib/')) {
        // Files in src/lib/ are at the same level
        content = content.replace(/from ['"]@\/lib\//g, 'from "./');
        content = content.replace(/from ['"]@\/components\//g, 'from "../components/');
        content = content.replace(/from ['"]@\/app\//g, 'from "../app/');
        content = content.replace(/from ['"]@\//g, 'from "./');
        console.log(`Converted lib file: ${filePath}`);
    } else {
        console.log(`No conversion needed for: ${filePath}`);
    }
    
    fs.writeFileSync(filePath, content);
}

// Find and convert all TypeScript/JavaScript files
const files = findFiles('src', ['.ts', '.tsx', '.js', '.jsx']);
files.forEach(convertImports);

console.log('✅ Import conversion completed!');
EOF

cd .build-temp
node convert_imports.js

# Debug: Check file structure after conversion
echo "🔍 Checking file structure after conversion..."
echo "Contents of src/ directory:"
ls -la src/
echo "Contents of src/lib/ directory:"
ls -la src/lib/
echo "Contents of src/components/ directory:"
ls -la src/components/

# Debug: Check a converted file
echo "🔍 Checking converted page.tsx:"
head -10 src/app/page.tsx

# Build with converted imports
echo "🏗️ Building with converted imports..."
npm run build

# Copy build output back to parent directory
echo "📦 Copying build output..."
cp -r .next ../.next
cp -r out ../out 2>/dev/null || true

# Go back to parent directory
cd ..

# Clean up temp directory
rm -rf .build-temp

# Verify the build output exists
echo "🔍 Verifying build output..."
ls -la .next/
if [ -f ".next/BUILD_ID" ]; then
    echo "✅ BUILD_ID found: $(cat .next/BUILD_ID)"
    echo "✅ Build completed successfully!"
else
    echo "❌ BUILD_ID not found!"
    echo "❌ Build failed!"
    exit 1
fi
