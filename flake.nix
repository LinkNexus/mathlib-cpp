{
  description = "mathlib-cpp: C++23 header-only math library";

  # Pinned, not tracking a branch: the point of a project shell is that
  # its toolchain only moves when someone moves it. Same revision as the
  # machine's system flake, so everything here is already in the store.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/8edc0c72e3a38faf5434e40d1f19431125fe4b30";

  outputs = {
    self,
    nixpkgs,
  }: let
    forEachSystem =
      nixpkgs.lib.genAttrs
      ["aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux"];
  in {
    devShells = forEachSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      default = pkgs.mkShell {
        nativeBuildInputs = [
          # `c++` with C++23 support -- mkShell supplies the host stdenv's
          # compiler. compile_commands.json comes from CMake itself
          # (CMAKE_EXPORT_COMPILE_COMMANDS), so bear isn't needed here.
          pkgs.cmake
          pkgs.ninja
        ];
      };
    });
  };
}
