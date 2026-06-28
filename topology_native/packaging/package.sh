#!/usr/bin/env bash
# Собирает release-бинарь и упаковывает в macOS .app + распространяемый .zip.
# Требует: rustup/cargo на PATH (source "$HOME/.cargo/env"), Xcode CLT.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"   # каталог topology_native/
NAME="TopologyNative"
VER="0.1.0"
APP="$HERE/dist/$NAME.app"
BIN="$HERE/target/release/topology_native"

echo "[1/4] cargo build --release"
cargo build --release --manifest-path "$HERE/Cargo.toml"

echo "[2/4] сборка $NAME.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$HERE/packaging/Info.plist" "$APP/Contents/Info.plist"
cp "$BIN" "$APP/Contents/MacOS/topology_native"
chmod +x "$APP/Contents/MacOS/topology_native"

echo "[3/4] ad-hoc codesign (нужно на arm64)"
codesign --force --deep --sign - "$APP"
codesign --verify --verbose=1 "$APP" || true

echo "[4/5] zip"
ZIP="$HERE/dist/$NAME-$VER-macos-arm64.zip"
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"

echo "[5/5] dmg (с симлинком на /Applications для drag-install)"
DMG="$HERE/dist/$NAME-$VER-macos-arm64.dmg"
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "Topology Native" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"

echo "готово:"
echo "  $APP"
echo "  $ZIP"
echo "  $DMG"
