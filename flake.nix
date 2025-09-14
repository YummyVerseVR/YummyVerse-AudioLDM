{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config. allowUnfree = true;
        config.cudaSupport = true;
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          pkg-config
          openssl
          cargo
          rustup
          poetry
          llvmPackages_14.libllvm
          python310
          # nvidia-docker
          cudatoolkit
          cudaPackages.cudnn
          uv
        ];
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [
          pkg-config
          openssl
          cargo
          rustup
          llvmPackages_14.libllvm
          stdenv.cc.cc.lib
          # nvidia-docker
          cudatoolkit
          cudaPackages.cudnn
          zlib
          libxcrypt
        ]);
        shellHook = ''
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH
        '';
      };
    };
}

