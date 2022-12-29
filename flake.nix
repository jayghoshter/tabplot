{
  inputs = { nixpkgs.url = "github:nixos/nixpkgs/67f49b2a3854e8b5e3f9df4422225daa0985f451"; };
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
  flake-utils.lib.eachDefaultSystem (system:
  let 
    pkgs = nixpkgs.legacyPackages.${system}; 
    # dir = ''/home/jayghoshter/dev/tools/chromoo/'';

    mach-nix = import (builtins.fetchGit {
      url = "https://github.com/DavHau/mach-nix";
      ref = "refs/tags/3.5.0";
    }) {};

    in 
    {

      defaultPackage = pkgs.python3Packages.buildPythonPackage{
        pname = "tabplot";
        version = "0.1";

        src = ./.;

        mnpyreq = mach-nix.mkPython {
          requirements = builtins.readFile ./requirements.txt;
        };

        propagatedBuildInputs = with pkgs; [
          mnpyreq
        ];

        doCheck = false;
      };


      devShell = pkgs.mkShell rec {
            name = "tabplot";

            mnpyreq = mach-nix.mkPython {
              requirements = builtins.readFile ./requirements.txt;
            };

            buildInputs = with pkgs; [
              mnpyreq
              git
              which
            ];

            shellHook = ''
                # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
                # See https://pip.pypa.io/en/stable/user_guide/#environment-variables.
                export PIP_PREFIX=$(pwd)/_build/pip_packages
                export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
                export PYTHONPATH="$(pwd):$PYTHONPATH"
                export PATH="$(pwd)/bin:$PATH"
                unset SOURCE_DATE_EPOCH
                '';

        };
      });
}
