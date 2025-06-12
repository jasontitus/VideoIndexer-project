#!/usr/bin/env bash
#
# build_and_sign.sh
# This script performs a full, clean build and/or sign of the VideoIndexer app.
# It is split into two stages: 'build' and 'sign'.
#
# USAGE:
#   1. Run the slow build process ONCE:
#      ./build_and_sign.sh build
#
#   2. Run the fast signing process repeatedly to debug:
#      ./build_and_sign.sh sign
# --------------------------------------------------------------------------
set -uo pipefail

# --- Configuration --------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ENV_NAME="video-indexer-build"
ENV_YAML_PATH="$SCRIPT_DIR/environment.yml"
APP_STRUCTURE_DIR="$SCRIPT_DIR/app_structure"
APP_SOURCE_DIR="$SCRIPT_DIR/src"
APP_BUILD_DIR="$SCRIPT_DIR/build"
FINAL_APP_PATH="$APP_BUILD_DIR/VideoIndexer.app"
SIGNING_IDENTITY="Developer ID Application: Jason Titus (A6G8H8NGAM)"
ENTITLEMENTS_PATH="$SCRIPT_DIR/entitlements.plist"

# ==========================================================================
# --- STAGE 1: BUILD -------------------------------------------------------
# ==========================================================================
build_app() {
    echo "▶ Starting Stage 1: Build"

    # --- Pre-flight Checks ----------------------------------------------------
    echo "▶ Setting up build directory: $APP_BUILD_DIR"
    rm -rf "$APP_BUILD_DIR"
    mkdir -p "$APP_BUILD_DIR"

    # --- Create Correct environment.yml ---------------------------------------
    echo "▶ Generating a clean environment.yml file..."
    cat > "$ENV_YAML_PATH" << EOF
name: video-indexer-build
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - ffmpeg=6.1.1
  - conda-pack=0.7.1
  - pip
EOF

    if [[ "$(uname -m)" != "arm64" ]]; then
      echo "❌ This script must be run on an Apple Silicon (arm64) Mac."
      exit 1
    fi

    if [ -d "$FINAL_APP_PATH" ]; then
        echo "▶ Removing old application bundle at $FINAL_APP_PATH"
        rm -rf "$FINAL_APP_PATH"
    fi

    if ! command -v conda &> /dev/null; then
        echo "❌ Conda is not found. Please make sure Conda is installed and in your PATH."
        exit 1
    fi

    # --- 1. Create Conda Environment from YAML --------------------------------
    echo "▶ Creating new Conda environment '$ENV_NAME' from $ENV_YAML_PATH..."
    export CONDA_SUBDIR=osx-arm64
    export ARCHFLAGS="-arch arm64"
    conda env remove --name "$ENV_NAME" --yes || true
    conda env create -f "$ENV_YAML_PATH"

    # --- 2. Install Pip Dependencies ------------------------------------------
    echo "▶ Installing Pip dependencies..."
    conda run -n "$ENV_NAME" pip install -r "$SCRIPT_DIR/requirements.txt"

    # --- 4. Assemble the Application Bundle -----------------------------------
    echo "▶ Assembling new application bundle at $FINAL_APP_PATH..."
    CONTENTS_DIR="$FINAL_APP_PATH/Contents"
    RESOURCES_DIR="$CONTENTS_DIR/Resources"
    MACOS_DIR="$CONTENTS_DIR/MacOS"
    mkdir -p "$RESOURCES_DIR" "$MACOS_DIR"

    ditto --norsrc "$APP_STRUCTURE_DIR/MacOS/VideoIndexer" "$MACOS_DIR/VideoIndexer"
    ditto --norsrc "$APP_STRUCTURE_DIR/Info.plist" "$CONTENTS_DIR/Info.plist"
    ditto --norsrc "$APP_STRUCTURE_DIR/PkgInfo" "$CONTENTS_DIR/PkgInfo"
    ditto --norsrc "$APP_STRUCTURE_DIR/Resources/VideoIndexer.icns" "$RESOURCES_DIR/VideoIndexer.icns"

    echo "▶ Copying application scripts and assets..."
    rsync -av "$APP_SOURCE_DIR/" "$RESOURCES_DIR/"

    # --- 5. Copy and Prepare the New Environment ------------------------------
    echo "▶ Locating Conda environment..."
    CONDA_ENV_PATH=$(conda run -n "$ENV_NAME" which python | xargs dirname | xargs dirname)
    echo "▶ Found environment at: $CONDA_ENV_PATH"

    FINAL_ENV_DIR="$RESOURCES_DIR/video-indexer"
    echo "▶ Copying full environment to the bundle at $FINAL_ENV_DIR..."
    mkdir -p "$FINAL_ENV_DIR"
    rsync -av --exclude 'pkgs' --exclude 'conda-meta' "$CONDA_ENV_PATH/" "$FINAL_ENV_DIR/"

    echo "🔧 Relocating environment..."
    "$FINAL_ENV_DIR/bin/conda-unpack"

    echo "✅ Stage 1: Build complete. App is assembled at $FINAL_APP_PATH"
    echo "▶ Now run './build_and_sign.sh sign' to attempt signing."
}

# ==========================================================================
# --- STAGE 2: SIGN --------------------------------------------------------
# ==========================================================================
sign_app() {
    echo "▶ Starting Stage 2: Sign"

    if [ ! -d "$FINAL_APP_PATH" ]; then
        echo "❌ Application bundle not found at $FINAL_APP_PATH"
        echo "▶ Please run './build_and_sign.sh build' first."
        exit 1
    fi

    # --- Final Cleaning (User-Discovered Solution) ------------------------
    echo "▶ Performing final clean based on successful manual signing..."
    echo "  - Step 1: Deleting any broken symbolic links..."
    find "$FINAL_APP_PATH" -type l ! -exec test -e {} \; -delete

    echo "  - Step 2: Thinning universal binaries to arm64-only..."
    # This command is structured to be robust. It will not cause the script
    # to exit if grep finds no matches.
    UNIVERSAL_FILES_TO_THIN=$(find "$FINAL_APP_PATH" -name "*.so" -exec file {} + | grep "universal binary" | grep "x86_64" | cut -d: -f1 || true)

    if [[ -n "$UNIVERSAL_FILES_TO_THIN" ]]; then
        echo "$UNIVERSAL_FILES_TO_THIN" | while read -r file_path; do
            echo "    - Thinning ${file_path##*/}"
            lipo -thin arm64 "$file_path" -output "$file_path"
        done
    else
        echo "    - No universal binaries found to thin."
    fi

    echo "  - Step 3: Recursively clearing all extended attributes..."
    xattr -rc "$FINAL_APP_PATH"

    # --- Signing (Inside-Out) ---------------------------------------------
    echo "▶ Performing robust, inside-out signing..."

    # 1. Sign all dynamic libraries and bundles with all entitlements
    echo "  - Signing .so and .dylib files..."
    find "$FINAL_APP_PATH" -type f \( -name "*.so" -o -name "*.dylib" \) -exec codesign --force --sign "$SIGNING_IDENTITY" --options runtime --timestamp --entitlements "$ENTITLEMENTS_PATH" {} \;

    # 2. Sign all other executables with all entitlements
    echo "  - Signing all other executables..."
    find "$FINAL_APP_PATH/Contents" -type f -perm +111 -exec codesign --force --sign "$SIGNING_IDENTITY" --options runtime --timestamp --entitlements "$ENTITLEMENTS_PATH" {} \;

    # 3. Sign the main application bundle itself, with all entitlements
    echo "  - Signing the main application bundle..."
    codesign --force --sign "$SIGNING_IDENTITY" --options runtime --timestamp --entitlements "$ENTITLEMENTS_PATH" -vvvv "$FINAL_APP_PATH"

    # --- Compression ------------------------------------------------------
    echo "▶ Compressing the application bundle for notarization..."
    APP_DIR=$(dirname "$FINAL_APP_PATH")
    APP_NAME=$(basename "$FINAL_APP_PATH")
    ZIP_NAME="${APP_NAME%.app}-mac.zip"

    # Remove old zip file if it exists
    if [ -f "$APP_DIR/$ZIP_NAME" ]; then
        echo "  - Removing old archive: $ZIP_NAME"
        rm "$APP_DIR/$ZIP_NAME"
    fi
    
    echo "  - Creating new archive at $APP_DIR/$ZIP_NAME"
    (
        cd "$APP_DIR"
        zip -ry9 "$ZIP_NAME" "$APP_NAME"
    )

    echo "✅ Signing and compression complete. Ready for notarization."
    echo "▶ Now run './build_and_sign.sh notarize'"
}

# ==========================================================================
# --- STAGE 3: NOTARIZE ----------------------------------------------------
# ==========================================================================
notarize_app() {
    echo "▶ Starting Stage 3: Notarize"

    # --- User Configuration ---------------------------------------------------
    # You must fill these in before running.
    # 1. Your Apple ID email
    APPLE_ID="jasontitus@mac.com"
    # 2. An app-specific password generated from https://appleid.apple.com
    APP_SPECIFIC_PASSWORD="codv-gvdk-woeq-xpbq"
    # 3. Your Team ID from https://developer.apple.com/account/#/membership/
    TEAM_ID="A6G8H8NGAM"
    # --------------------------------------------------------------------------

    SIGNED_ZIP_PATH="$APP_BUILD_DIR/VideoIndexer-mac.zip"
    KEYCHAIN_PROFILE="VideoIndexerNotaryProfile"

    if [ "$APPLE_ID" == "developer@example.com" ] || [ "$APP_SPECIFIC_PASSWORD" == "abcd-efgh-ijkl-mnop" ] || [ "$TEAM_ID" == "YOUR_TEAM_ID" ]; then
        echo "❌ Notarization credentials are not set. Please edit the 'notarize_app' function in this script and fill in your details."
        exit 1
    fi

    echo "▶ Checking for signed zip file at: $SIGNED_ZIP_PATH"
    if [ ! -f "$SIGNED_ZIP_PATH" ]; then
        echo "❌ Signed zip file not found at $SIGNED_ZIP_PATH"
        echo "▶ Please run './build_and_sign.sh sign' first."
        exit 1
    fi
    echo "   Zip file found."

    echo "▶ Submitting for notarization (this may take several minutes)..."
    
    # Enable command tracing
    set -x

    # Submit the app and wait for the result
    SUBMISSION_RESPONSE=$(xcrun notarytool submit "$SIGNED_ZIP_PATH" --keychain-profile "$KEYCHAIN_PROFILE" --wait --output-format json)
    SUBMISSION_EXIT_CODE=$?
    
    # Disable command tracing
    set +x
    
    echo "▶ Submission command finished with exit code: $SUBMISSION_EXIT_CODE"
    echo "▶ Submission command has finished. Raw response:"
    echo "$SUBMISSION_RESPONSE"

    SUBMISSION_ID=$(echo "$SUBMISSION_RESPONSE" | grep -o '"id": "[^"]*"' | cut -d'"' -f4)
    STATUS=$(echo "$SUBMISSION_RESPONSE" | grep -o '"status": "[^"]*"' | cut -d'"' -f4)

    echo "▶ Parsed Submission ID: $SUBMISSION_ID"
    echo "▶ Parsed Status: $STATUS"

    if [ -z "$STATUS" ]; then
        echo "❌ Could not parse notarization status from response. Raw response was:"
        echo "$SUBMISSION_RESPONSE"
        exit 1
    fi

    if [ "$STATUS" != "Accepted" ]; then
        echo "❌ Notarization FAILED. Status: $STATUS"
        echo "   Submission ID: $SUBMISSION_ID"
        echo "   Attempting to fetch logs..."
        
        # Enable command tracing for the log command
        set -x
        xcrun notarytool log "$SUBMISSION_ID" --keychain-profile "$KEYCHAIN_PROFILE"
        set +x

        exit 1
    fi

    echo "✅ Notarization successful!"
    echo "▶ Stapling notarization ticket to the application..."

    # Unzip the application so we can staple the ticket to it
    unzip -o "$SIGNED_ZIP_PATH" -d "$APP_BUILD_DIR"
    
    # Staple the ticket
    xcrun stapler staple "$FINAL_APP_PATH"
    
    echo "▶ Re-zipping the final, notarized application..."
    (
        cd "$APP_BUILD_DIR"
        zip -ry9 "VideoIndexer-mac.zip" "VideoIndexer.app"
    )

    echo "  - Moving final notarized archive to Desktop..."
    mv "$APP_BUILD_DIR/VideoIndexer-mac.zip" "$HOME/Desktop/VideoIndexer-mac.zip"

    echo "🎉 Notarization and stapling complete!"
    echo "▶ Run './build_and_sign.sh verify' to confirm Gatekeeper acceptance."
}

# ==========================================================================
# --- STAGE 4: VERIFY ------------------------------------------------------
# ==========================================================================
verify_app() {
    echo "▶ Starting Stage 4: Verify"
    echo "▶ Verifying the final application bundle with spctl..."

    if [ ! -d "$FINAL_APP_PATH" ]; then
        echo "❌ Application bundle not found at $FINAL_APP_PATH"
        echo "▶ Please run './build_and_sign.sh build' first."
        exit 1
    fi

    spctl --assess --verbose=4 --type execute "$FINAL_APP_PATH"
}


# --- Main Logic -----------------------------------------------------------
if [[ "$#" -ne 1 ]] || [[ "$1" != "build" && "$1" != "sign" && "$1" != "notarize" && "$1" != "verify" ]]; then
    echo "Usage: $0 [build|sign|notarize|verify]"
    echo "NOTE: Before running 'notarize' for the first time, you must run './store_credentials.sh'"
    exit 1
fi

if [[ "$1" == "build" ]]; then
    build_app
elif [[ "$1" == "sign" ]]; then
    sign_app
elif [[ "$1" == "notarize" ]]; then
    notarize_app
elif [[ "$1" == "verify" ]]; then
    verify_app
fi 