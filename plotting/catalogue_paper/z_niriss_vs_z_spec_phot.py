"""
Compare the NIRISS redshifts to existing spectroscopic redshifts.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth / 1.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        # hspace=0.,
        # wspace=0.,
    )
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

    # secure = full_cat["Z_FLAG_ALL"] >= 9
    # tentative = (full_cat["Z_FLAG_ALL"] == 7) | (full_cat["Z_FLAG_ALL"] == 6)

    secure = full_cat["Z_FLAG"] >= 4
    tentative = full_cat["Z_FLAG"] == 3

    spec = np.zeros(len(full_cat), dtype=bool)
    # spec[np.isfinite(full_cat["zspec"])] = True
    spec[np.isfinite(full_cat["zmed_prev"])] = True
    phot = np.zeros(len(full_cat), dtype=bool)
    # phot[np.isfinite(full_cat["zphot"]) & ~spec] = True
    phot[
        np.isfinite(full_cat["zphot_astrodeep"])
        & (~spec)
        & (full_cat["flag_astrodeep"] != 100028)
    ] = True
    # print (np.nansum(phot))
    no_z = np.zeros(len(full_cat), dtype=bool)
    no_z[~phot & ~spec] = True
    # phot = (np.isfinite(full_cat["zphot"])) & (~(np.isfinite(full_cat["zspec"]) & full_cat["zspec"].mask))
    print(np.sum(spec), np.sum(phot))

    # print (full_cat["flag_astrodeep"][5:10])

    print(
        np.sum(secure),
        np.sum(secure & spec),
        np.sum(secure & phot),
        np.sum(secure & no_z),
    )
    print(
        np.sum(tentative),
        np.sum(tentative & spec),
        np.sum(tentative & phot),
        np.sum(tentative & no_z),
    )
    print(
        np.sum(secure | tentative),
        np.sum((secure | tentative) & spec),
        np.sum((secure | tentative) & phot),
        np.sum((secure | tentative) & no_z),
    )

    print()

    # exit()

    # z_range = np.linspace(
    #     0, np.nanmax([full_cat["zphot"][secure], full_cat["Z_EST_ALL"][secure]]) * 1.05, 2
    # )
    z_range = np.linspace(0, 8.5, 2)
    print(z_range)
    for a in axs.flatten():
        a.plot(z_range, z_range, c="k", linestyle=":", linewidth=1, zorder=-1)

    for i, (z_name, z_cat_name, idx) in enumerate(
        zip(
            [r"$z_{\rm{spec}}$", r"$z_{\rm{phot}}$"],
            ["zmed_prev", "zphot_astrodeep"],
            [spec, phot],
        )
    ):
        axs[i].scatter(
            full_cat["Z_NIRISS"][secure & idx],
            full_cat[z_cat_name][secure & idx],
            # full_cat["MAG_AUTO"][secure],
            color="purple",
            alpha=0.7,
            # edgecolor="none",
            s=7,
            # bins=z_bins,
            label=z_name,
        )
        scatter = np.nanstd(
            (full_cat["Z_NIRISS"][secure & idx] - full_cat[z_cat_name][secure & idx])
        )
        axs[i].text(
            0.1,
            0.9,
            rf"$\sigma_{{z}}={scatter:.2f}$",
            va="top",
            transform=axs[i].transAxes,
        )

        print(
            full_cat["NUMBER", "Z_NIRISS"][
                (secure & idx)
                & (np.abs(full_cat["Z_NIRISS"] - full_cat[z_cat_name]) > 0.25)
            ]
        )

    # ax.hist(
    #     # full_cat["zspec"][secure],
    #     full_cat["zphot"][secure] - full_cat["Z_EST_ALL"][secure],
    #     # full_cat["MAG_AUTO"][secure],
    #     color="purple",
    #     # alpha=.7,
    #     # edgecolor="none",
    #     # s=7,
    #     bins=np.arange(-0.025, 0.025, 0.001),
    # )
    # print (full_cat["ID"][(full_cat["zspec"]>=7) & secure])
    # ax.scatter(
    #     full_cat["Z_EST_ALL"][tentative],
    #     full_cat["MAG_AUTO"][tentative],
    #     color="k",
    #     facecolor="none",
    #     marker="o",
    #     alpha=.7,
    #     s=4,
    #     linewidth=0.5,
    # )
    # ax.set_xlim(z_bins[0], z_bins[-1])
    # y_lims = ax.get_ylim()

    # h_alpha_lims = [
    #     [0.543, 0.954],
    #     [1.026, 1.545],
    #     [1.667, 2.391],
    # ]
    # OIII_lims = [
    #     [1.032, 1.574],
    #     [1.668, 2.353],
    #     [2.513, 3.466],
    # ]
    # OII_lims = [
    #     [1.717,2.441],
    #     [2.567,3.482],
    #     [3.696,4.970],
    # ]
    # for i, l in enumerate(h_alpha_lims):
    #     ax.fill_betweenx(y_lims, *l, color="r", alpha=0.15, edgecolor="none", label=r"H$\alpha$" if i==0 else None)
    # for i, l in enumerate(OIII_lims):
    #     ax.fill_betweenx(y_lims, *l, color="g", alpha=0.15, edgecolor="none", label=r"[O\,\textsc{iii}]" if i==0 else None)
    # for i, l in enumerate(OII_lims):
    #     ax.fill_betweenx(y_lims, *l, color="b", alpha=0.15, edgecolor="none", label=r"[O\,\textsc{ii}]" if i==0 else None)

    # ax.set_ylim(y_lims)

    # # cat_names = ["grizli_photz_matched.fits"]

    # # catalogue_path = catalogue_dir / "grizli_photz_matched.fits"

    # v1_cat = Table.read(catalogue_dir / "compiled_catalogue_v1.fits")

    # z_min = 0
    # z_max = 3
    # z_range = np.linspace(z_min, z_max, 100)

    # for k, v in line_dict.items():
    #     ax.plot(z_range, (1 + z_range) * v)

    # for k, v in niriss_info.items():
    #     ax.fill_between(z_range, v[0], v[1], color="k", alpha=0.4, edgecolor="none")

    # ax.set_ylim(9000, 23500)
    # ax.set_xlim(z_min, z_max)
    # ax.axvline(1.34)
    # ax.axvline(1.98)

    # # hist = partial(plot_utils.plot_hist, bins=np.arange(16.5, 33.5, 0.5), ax=ax)
    # # # print (np.nanmin(v1_cat["MAG_AUTO"]), np.nanmax(v1_cat["MAG_AUTO"])

    # # hist(v1_cat["MAG_AUTO"], label="Full Sample")
    # # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] > 0], ax=ax, label="Extracted")
    # # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 4], ax=ax, label="First Pass")
    # # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 5], ax=ax, label="Placeholder")

    # # ax.set_ylabel(r"$m_{\rm{F200W}}$")
    # axs[1].semilogy()
    # axs[1].semilogx()
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    # axs[1].set_xticks(
    z_ticks = np.concatenate([np.arange(0.1, 1, 0.1), np.arange(1.0, 9.0, 1)])
    z_ticks_labels = [
        "0.1",
        "0.2",
        "",
        "",
        "0.5",
        "",
        "",
        "",
        "",
        "1.0",
        "2.0",
        "",
        "",
        "5.0",
        "",
        "",
        "",
    ]
    # axs[1].set_xticklabels([])
    axs[1].set_xticks(z_ticks, z_ticks_labels, minor=True)
    axs[1].set_yticks(z_ticks, z_ticks_labels, minor=True)
    axs[1].set_xticks([0.1, 1.0], ["0.1", "1.0"], minor=False)
    axs[1].set_yticks([0.1, 1.0], ["0.1", "1.0"], minor=False)
    axs[1].set_xlabel(r"$z_{\rm{phot}}$")
    axs[0].set_xlabel(r"$z_{\rm{spec}}$")
    axs[0].set_ylabel(r"$z_{\rm{NIRISS}}$")
    # axs[1].set_ylabel(r"$z_{\rm{NIRISS}}$")
    # # ax.set_ylabel(r"Number of Objects")
    # # ax.legend()

    axs[1].set_xlim([0.08, 10])
    axs[1].set_ylim([0.08, 10])
    #
    # plt.subplots_adjust(wspace=0, hspace=0)

    # plt.savefig(save_dir / "z_niriss_vs_z_spec_phot.pdf")
    # for k, v in line_dict.items():
    #     for k_n, v_n in niriss_info.items():
    #         # low = v_n[0]/v -1
    #         print(f"{k}: {k_n}={v_n[0]/v-1:.3f},{v_n[1]/v-1:.3f}")

    plt.show()
