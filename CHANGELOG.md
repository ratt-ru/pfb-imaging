# Changelog

All notable changes to pfb-imaging are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### CI

- Align update-cabs workflow with hip-cargo pattern
- **deps**: Bump docker/build-push-action from 6 to 7 (#187) ([#187](https://github.com/ratt-ru/pfb-imaging/pull/187))
- **deps**: Bump actions/cache from 4 to 5 (#173) ([#173](https://github.com/ratt-ru/pfb-imaging/pull/173))
- **deps**: Bump actions/checkout from 4 to 6 (#169) ([#169](https://github.com/ratt-ru/pfb-imaging/pull/169))

### Dependencies

- Update distributed requirement from <2026.2.0 to <2026.4.0 (#223)
- Update uv-build requirement (#222)
- Update distributed requirement from <2026.2.0 to <2026.4.0 (#216)

### Documentation

- Update copilot-instructions container-workflow block
- Update README container-images section
- Update CLAUDE.md image-resolution section

### Fixed

- Fix Nyquist cell size computation

### Miscellaneous

- Use hip-cargo pattern in tbump before_commit hooks
- Add __version__ literal for tbump
- Move container image to hip.cargo entry point

### Other

- Import PsiOperatorProtocol from opt module
- New hip-cargo workflow for container fallback etc
- Create _container_image.py file and add get_container_image() calls to CLI functions
- Bump aiohttp from 3.13.3 to 3.13.4 (#225)
- Bump pygments from 2.19.2 to 2.20.0 (#224)
- Bump cryptography from 46.0.5 to 46.0.6 (#221)
- Auto ff to latest hc workflow
- Add omegaconf dependency
- Depend on hip-cargo 0.1.7
- Remove help panel for non-stimela params for now as they cause roundtrip to fail
- Bump requests from 2.32.5 to 2.33.0 (#219)
- Add rich help panels everywhere
- Make cube_to_fits functional in hci app
- Fix #217: Fix (hopefully) PSF fitting code and add tests (#218) ([#218](https://github.com/ratt-ru/pfb-imaging/pull/218))
- Fix flux suppression issue in `hci`  (#215) ([#215](https://github.com/ratt-ru/pfb-imaging/pull/215))
- Remove dirt comment (#208) ([#208](https://github.com/ratt-ru/pfb-imaging/pull/208))
- Useapptoupdatecabs2 (#207) ([#207](https://github.com/ratt-ru/pfb-imaging/pull/207))
- Add ability to dispatch update cabs manually [skip ci] (#205) ([#205](https://github.com/ratt-ru/pfb-imaging/pull/205))
- Fix update cabs workflow (#204) ([#204](https://github.com/ratt-ru/pfb-imaging/pull/204))
- Consolidate dependabot PRs (#202) ([#202](https://github.com/ratt-ru/pfb-imaging/pull/202))
- Transition to `hip-cargo` format (#176) ([#176](https://github.com/ratt-ru/pfb-imaging/pull/176))
- Fix signs of injected transient (#172) ([#172](https://github.com/ratt-ru/pfb-imaging/pull/172))

### Testing

- Test commit


## [0.0.8] - 2025-10-21

### Other

- Keep dims when converting model to Stokes in stokes2im (#168) ([#168](https://github.com/ratt-ru/pfb-imaging/pull/168))
- Tmpdirout (#166) ([#166](https://github.com/ratt-ru/pfb-imaging/pull/166))
- Make temp-dir an output
- Add temp-dir for synchronizer, remove upper padding limit, make output name explicit (#165) ([#165](https://github.com/ratt-ru/pfb-imaging/pull/165))


## [0.0.7] - 2025-10-06

### CI

- **deps**: Bump actions/setup-python from 5 to 6 (#152) ([#152](https://github.com/ratt-ru/pfb-imaging/pull/152))
- **deps**: Bump abatilo/actions-poetry from 3 to 4 (#134) ([#134](https://github.com/ratt-ru/pfb-imaging/pull/134))
- **deps**: Bump pypa/gh-action-pypi-publish from 1.12.2 to 1.12.4 (#133) ([#133](https://github.com/ratt-ru/pfb-imaging/pull/133))

### Fixed

- Fix uninitialised variables in restore (#154) ([#154](https://github.com/ratt-ru/pfb-imaging/pull/154))
- Fix reviewers in dependabot (#135) ([#135](https://github.com/ratt-ru/pfb-imaging/pull/135))

### Other

- Cubemean (#163) ([#163](https://github.com/ratt-ru/pfb-imaging/pull/163))
- Set hci ext according to stack (#162) ([#162](https://github.com/ratt-ru/pfb-imaging/pull/162))
- Use .zarr or .fds extension depending on whether stack is set in hci (#161) ([#161](https://github.com/ratt-ru/pfb-imaging/pull/161))
- Prefer `obs-label` to `obslabel` (#160) ([#160](https://github.com/ratt-ru/pfb-imaging/pull/160))
- Add OBSLABEL to fits header in stacked cube attributes (#158) ([#158](https://github.com/ratt-ru/pfb-imaging/pull/158))
- Round coords when create dummy ds to avoid roudning errors. change chan_widths -> channel_width and remove attribute (#156) ([#156](https://github.com/ratt-ru/pfb-imaging/pull/156))
- Tweak logging (#149) ([#149](https://github.com/ratt-ru/pfb-imaging/pull/149))
- Bump pypa/gh-action-pypi-publish in /.github/workflows (#147) ([#147](https://github.com/ratt-ru/pfb-imaging/pull/147))
- Set beam type to URI (#148) ([#148](https://github.com/ratt-ru/pfb-imaging/pull/148))
- Add chan_widths (actually band widths) to stacked cube (#146) ([#146](https://github.com/ratt-ru/pfb-imaging/pull/146))
- Separate wsum and beam weights in hci worker (#144) ([#144](https://github.com/ratt-ru/pfb-imaging/pull/144))
- Simple transient simulation functionality (#142) ([#142](https://github.com/ratt-ru/pfb-imaging/pull/142))
- Cast float to array in make_dummy_dataset (#141) ([#141](https://github.com/ratt-ru/pfb-imaging/pull/141))
- Add on the fly rephasing and beam interpolation/reprojection to hci worker (#139) ([#139](https://github.com/ratt-ru/pfb-imaging/pull/139))
- Simplify degrid (#138) ([#138](https://github.com/ratt-ru/pfb-imaging/pull/138))


## [0.0.6] - 2025-07-15

### Other

- V0.0.6 dev (#132) ([#132](https://github.com/ratt-ru/pfb-imaging/pull/132))


## [0.0.5] - 2024-12-13

### Other

- Try again skip ci (#129) ([#129](https://github.com/ratt-ru/pfb-imaging/pull/129))
- Tmp (#128) ([#128](https://github.com/ratt-ru/pfb-imaging/pull/128))
- Tmp (#127) ([#127](https://github.com/ratt-ru/pfb-imaging/pull/127))
- Factions (#126) ([#126](https://github.com/ratt-ru/pfb-imaging/pull/126))
- V0.0.5 (#125) ([#125](https://github.com/ratt-ru/pfb-imaging/pull/125))
- V0.0.5 dev (#122) ([#122](https://github.com/ratt-ru/pfb-imaging/pull/122))
- Changed stimela dependency (#124) ([#124](https://github.com/ratt-ru/pfb-imaging/pull/124))
- Add pyproject.toml and relax dependencies (#117) ([#117](https://github.com/ratt-ru/pfb-imaging/pull/117))
- Clean up schema and dependencies  (#114) ([#114](https://github.com/ratt-ru/pfb-imaging/pull/114))
- Refactor to make all algorithms use the same formats, simplify parallelism, direct pre-conditioning and produce uniformly blurred images (#111) ([#111](https://github.com/ratt-ru/pfb-imaging/pull/111))


## [0.0.3] - 2024-05-02

### Fixed

- Fix div by zero in fitcleanbeam (#96) ([#96](https://github.com/ratt-ru/pfb-imaging/pull/96))

### Other

- Update deploy workflow (#103) ([#103](https://github.com/ratt-ru/pfb-imaging/pull/103))
- V0.0.2 dev (#102) ([#102](https://github.com/ratt-ru/pfb-imaging/pull/102))
- Set min counts to fraction of median (#100) ([#100](https://github.com/ratt-ru/pfb-imaging/pull/100))
- Add high cadence imaging worker and ability to turn frames into movies (#97) ([#97](https://github.com/ratt-ru/pfb-imaging/pull/97))
- Awskube (#91) ([#91](https://github.com/ratt-ru/pfb-imaging/pull/91))


## [0.0.1] - 2023-07-30

### CI

- Ci and pytest (#14) ([#14](https://github.com/ratt-ru/pfb-imaging/pull/14))

### Fixed

- Fix for parsing nworkers (#77) ([#77](https://github.com/ratt-ru/pfb-imaging/pull/77))

### Other

- Newtokens (#90) ([#90](https://github.com/ratt-ru/pfb-imaging/pull/90))
- Add [skip ci] option to workflow (#89) ([#89](https://github.com/ratt-ru/pfb-imaging/pull/89))
- Use actions/checkout v3 (#88) ([#88](https://github.com/ratt-ru/pfb-imaging/pull/88))
- Trigger CI on push with v* (#87) ([#87](https://github.com/ratt-ru/pfb-imaging/pull/87))
- Correct current version (#86) ([#86](https://github.com/ratt-ru/pfb-imaging/pull/86))
- Prepare for v0.0.2 (first public beta) (#84) ([#84](https://github.com/ratt-ru/pfb-imaging/pull/84))
- Depend on qcal main (#85) ([#85](https://github.com/ratt-ru/pfb-imaging/pull/85))
- Add time chunks to gain_chunks dict (#83) ([#83](https://github.com/ratt-ru/pfb-imaging/pull/83))
- Prep main
- Solarkat (#80) ([#80](https://github.com/ratt-ru/pfb-imaging/pull/80))
- Pdfromupdate (#76) ([#76](https://github.com/ratt-ru/pfb-imaging/pull/76))
- Revert "start pd from update if model is zero"
- Start pd from update if model is zero
- New psf convolve (#75) ([#75](https://github.com/ratt-ru/pfb-imaging/pull/75))
- Better FFT's (#74) ([#74](https://github.com/ratt-ru/pfb-imaging/pull/74))
- Refactor model + distributed deconvolution (#68) ([#68](https://github.com/ratt-ru/pfb-imaging/pull/68))
- Add support for single channel measurement sets (#66) ([#66](https://github.com/ratt-ru/pfb-imaging/pull/66))
- Mappings (#63) ([#63](https://github.com/ratt-ru/pfb-imaging/pull/63))
- Awskube (#62) ([#62](https://github.com/ratt-ru/pfb-imaging/pull/62))
- Wait for workers when using distributed scheduler (#61) ([#61](https://github.com/ratt-ru/pfb-imaging/pull/61))
- AWS kubernetes integration (#58) ([#58](https://github.com/ratt-ru/pfb-imaging/pull/58))
- Auto-threshold clean + Dockerfile (#57) ([#57](https://github.com/ratt-ru/pfb-imaging/pull/57))
- Additional tests and improved ss clean (#56) ([#56](https://github.com/ratt-ru/pfb-imaging/pull/56))
- Stimelation (#54) ([#54](https://github.com/ratt-ru/pfb-imaging/pull/54))
- Fix entry point specification in setup.py
- Compute in subiters (#53) ([#53](https://github.com/ratt-ru/pfb-imaging/pull/53))
- The rise of the workers [WIP] (#49) ([#49](https://github.com/ratt-ru/pfb-imaging/pull/49))
- Fix publish gh action (#47) ([#47](https://github.com/ratt-ru/pfb-imaging/pull/47))
- Release again (#46) ([#46](https://github.com/ratt-ru/pfb-imaging/pull/46))
- Add publich action (#45) ([#45](https://github.com/ratt-ru/pfb-imaging/pull/45))
- Spotless (#42) ([#42](https://github.com/ratt-ru/pfb-imaging/pull/42))
- Refactor minors (#39) ([#39](https://github.com/ratt-ru/pfb-imaging/pull/39))
- Spi fitter (#37) ([#37](https://github.com/ratt-ru/pfb-imaging/pull/37))
- Get master up to date (#34) ([#34](https://github.com/ratt-ru/pfb-imaging/pull/34))
- Ssclean (#26) ([#26](https://github.com/ratt-ru/pfb-imaging/pull/26))
-  Corrections to default args in test_pfb (#23) ([#23](https://github.com/ratt-ru/pfb-imaging/pull/23))
- L2 reweighting (#21) ([#21](https://github.com/ratt-ru/pfb-imaging/pull/21))
- Streamline test_pfb (#20) ([#20](https://github.com/ratt-ru/pfb-imaging/pull/20))
- Fits mask (#19) ([#19](https://github.com/ratt-ru/pfb-imaging/pull/19))
- Fits mask (#18) ([#18](https://github.com/ratt-ru/pfb-imaging/pull/18))
- Fix minor bug in make_mask (#17) ([#17](https://github.com/ratt-ru/pfb-imaging/pull/17))
- Project init (#16) ([#16](https://github.com/ratt-ru/pfb-imaging/pull/16))
- Setup ci (#13) ([#13](https://github.com/ratt-ru/pfb-imaging/pull/13))
- Initial commit


[Unreleased]: https://github.com/ratt-ru/pfb-imaging/compare/v0.0.8...HEAD
[0.0.8]: https://github.com/ratt-ru/pfb-imaging/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/ratt-ru/pfb-imaging/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/ratt-ru/pfb-imaging/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/ratt-ru/pfb-imaging/compare/v0.0.4...v0.0.5
[0.0.3]: https://github.com/ratt-ru/pfb-imaging/compare/v0.0.1...v0.0.3
[0.0.1]: https://github.com/ratt-ru/pfb-imaging/releases/tag/v0.0.1
