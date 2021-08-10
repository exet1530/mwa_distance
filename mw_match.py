import numpy as np
from scipy.spatial import distance
from astropy.table import Table
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
import calc_kcor

"""
    Thoughts: Top 10 appears robust against random errors - that's encouraging
    
    
    Should I reject anything that has a preference for a 1-component fit?
    Unclear - would the MW show clear evidence of two components? Maybe...
    This does make a fair difference to the top 10 so worth thinking about.
    Keep doing both for now
    
    Which variables are important? Mass seems the least informative. B/T seems the
    most informative, in that when excluded the distances shift the most
    
    Can I model the effect of dust on kinematics? Should be straightforward
    see Kregel and van der Kruit (2005) - fig 1.
    What about pops? Seems a lot harder...
"""

def calculate_similarities(errs = True,inc_ss=True):

    galaxy_data,mw_data = prep_data(inc_ss=inc_ss)
    
    pos_mw = [mw_data['Mstar'][0],
                mw_data['g-i'][0],
                mw_data['h_g'][0],
                mw_data['h_i'][0],
                mw_data['B/T'][0]
                ]
                  
    err_mw = [mw_data['Mstar'][1],
                mw_data['g-i'][1],
                mw_data['h_g'][1],
                mw_data['h_i'][1],
                mw_data['B/T'][1]
                ]
    
    distances = []
    for i in range(len(galaxy_data)):
        pos_gal = [galaxy_data['Mstar'][i],
                    galaxy_data['g-i'][i],
                    galaxy_data['h_g'][i],
                    galaxy_data['h_i'][i],
                    galaxy_data['B/T'][i]
                    ]
                    
        err_gal = [galaxy_data['Mstar_err'][i],
                    galaxy_data['g-i_err'][i],
                    galaxy_data['h_g_err'][i],
                    galaxy_data['h_i_err'][i],
                    galaxy_data['B/T_err'][i]
                    ]
        
        if errs == False:
            try:
                dist = distance.euclidean(pos_gal,pos_mw)
            except:
                dist = np.nan
        else:
        
            try:
                dist = calc_dist(pos_gal,pos_mw,err_gal,err_mw)
            except:
                dist = np.nan
        distances.append(dist)
        
    return galaxy_data,mw_data,distances

def calculate_similarities_twist(errs = True,twist=0.1,ntwist=100):

    galaxy_data,mw_data = prep_data()
    bestest = []
    
    for i in range(ntwist):
    
        pos_mw = [mw_data['Mstar'][0]*(np.random.randn()*twist)+1,
                    mw_data['g-i'][0]+(np.random.randn()*twist*0.5),
                    mw_data['h_g'][0]+(np.random.randn()*twist),
                    mw_data['h_i'][0]+(np.random.randn()*twist),
                    mw_data['B/T'][0]*(np.random.randn()*twist)+1]
                  
        err_mw = [mw_data['Mstar'][1],
                    mw_data['g-i'][1],
                    mw_data['h_g'][1],
                    mw_data['h_i'][1],
                    mw_data['B/T'][1]]
    
        distances = []
        for i in range(len(galaxy_data)):
            pos_gal = [galaxy_data['Mstar'][i],
                        galaxy_data['g-i'][i],
                        galaxy_data['h_g'][i],
                        galaxy_data['h_i'][i],
                        galaxy_data['B/T'][i]]
                    
            err_gal = [galaxy_data['Mstar_err'][i],
                        galaxy_data['g-i_err'][i],
                        galaxy_data['h_g_err'][i],
                        galaxy_data['h_i_err'][i],
                        galaxy_data['B/T_err'][i]]
        
            if errs == False:
                try:
                    dist = distance.euclidean(pos_gal,pos_mw)
                except:
                    dist = np.nan
            else:
        
                try:
                    dist = calc_dist(pos_gal,pos_mw,err_gal,err_mw,n_iter=1000)
                except:
                    dist = np.nan
            if np.isfinite(dist) == False:
                import code
                code.interact(local=dict(globals(),**locals()))
            distances.append(dist)
            
        inds = np.argsort(distances)
        best = galaxy_data['CATID'][inds][:10]
        bestest.append(best)
        
    return bestest

def prep_data(inc_ss=True):

    galaxy_data = load_galaxy_data(inc_ss=inc_ss)
    galaxy_data_clean = clean_galaxy_data(galaxy_data)
    
    mw_data = load_mw_data()
    mw_data = rdisk_BVR_to_ugriz(mw_data)
    
    mw_data['h_g'] = [np.log10(mw_data['h_g'][0]),mw_data['h_g'][1]/mw_data['h_g'][0]]
    mw_data['h_i'] = [np.log10(mw_data['h_i'][0]),mw_data['h_i'][1]/mw_data['h_i'][0]]

    
    normalised_galaxy_data = Table(galaxy_data_clean)
    normalised_mw_data = dict(mw_data)
    
    
    for quantity in ['Mstar','B/T','h_g','h_i','M_r','g-i']:
        normed_data,mn,sig = normalize(galaxy_data_clean[quantity])
        normalised_galaxy_data[quantity] = normed_data
        normalised_galaxy_data[quantity+'_err'] = normalised_galaxy_data[quantity+'_err']/sig
        normalised_mw_data[quantity] = [(normalised_mw_data[quantity][0]-mn)/sig,
                                        normalised_mw_data[quantity][1]/sig]
        
    ww = np.where((normalised_galaxy_data['h_g_err'] < 0) | (normalised_galaxy_data['h_g_err'] > 5))
    normalised_galaxy_data['h_g_err'][ww] = np.nanmedian(normalised_galaxy_data['h_g_err'])*5
    ww = np.where((normalised_galaxy_data['h_i_err'] < 0) | (normalised_galaxy_data['h_i_err'] > 5))
    normalised_galaxy_data['h_i_err'][ww] = np.nanmedian(normalised_galaxy_data['h_i_err'])*5
                                    
    return normalised_galaxy_data,normalised_mw_data

def cluster_dist_mod(id):

    # Return the distance modulus of the 8 sami clusters

    id = str(id)
    id = id[1:5]

    clusters = np.array(['0085','0119','0168','0442','0917','2399','3880','4038'])
    dist_mods = [36.74,36.25,36.29,36.52,36.62,36.88,36.90,35.30]

    w = np.where(id == clusters)[0]
    
    return dist_mods[w[0]]

def load_galaxy_data(inc_ss=True):

    # Read in various SAMI galaxy properties and organise into a single table
    # Rescale values to be comparable to the MW where necessary e.g. sizes -> kpc

    tab_sample = Table.read('/Users/nscott/Data/SAMI/Survey/sami_sel_20140911_v2.0_FINALobs.dat',
                                format='ascii')
    tab_sersic = Table.read('/Users/nscott/Data/SAMI/Survey/BDDecomp_SAMIv03/BDModelsv03.fits')
    
    tab_ex = Table.read('/Users/nscott/Data/SAMI/Survey/samiv2.0_GalacticExtinctionv02.fits')
    tab_ap = Table.read('/Users/nscott/Data/SAMI/Survey/samiv2.0_ApMatchedCatv03.fits')
    
    tab_sample_cluster = Table.read('/Users/nscott/Data/SAMI/Survey/ClustersCombined_V10_FINALobs.dat',
                                format='ascii')
    tab_sample_cluster = tab_sample_cluster[1:] # Removes duplicate initial line
    tab_sersic_cluster_r = Table.read('/Users/nscott/Data/SAMI/Survey/BDclustersSB/SDSS_ATLAS_rband.csv')
    tab_sersic_cluster_g = Table.read('/Users/nscott/Data/SAMI/Survey/BDclustersSB/SDSS_ATLAS_gband.csv')
    tab_sersic_cluster_i = Table.read('/Users/nscott/Data/SAMI/Survey/BDclustersSB/SDSS_ATLAS_iband.csv')
        
    tab_full = Table()

    ww = np.where(tab_sample['SAMI_OBS'] == 1)
    vv = np.where(tab_sample_cluster['observed_flag'] == 1)
    
    n_gama = len(tab_sample[ww])
    n_cluster = len(tab_sample_cluster[vv])
    n_galaxies = n_gama + n_cluster
    
    tab_full['CATID'] = np.zeros(n_galaxies)
    tab_full['CATID'][:n_gama] = tab_sample['CATID'][ww]
    tab_full['CATID'][n_gama:] = tab_sample_cluster['CATAID'][vv]
    tab_full['Mstar'] = np.zeros(n_galaxies)
    tab_full['Mstar'][:n_gama] = tab_sample['Mstar'][ww]
    tab_full['Mstar'][n_gama:] = tab_sample_cluster['Mstar'][vv]
    tab_full['Mstar_err'] = np.ones(n_galaxies)*0.1
    
    tab_full['B/T'] = np.zeros(n_galaxies)
    tab_full['h_g'] = np.zeros(n_galaxies)
    tab_full['h_i'] = np.zeros(n_galaxies)
    tab_full['Re'] = np.zeros(n_galaxies)

    tab_full['B/T_err'] = np.zeros(n_galaxies)
    tab_full['h_g_err'] = np.zeros(n_galaxies)
    tab_full['h_i_err'] = np.zeros(n_galaxies)
    
    tab_full['M_r'] = np.zeros(n_galaxies)
    tab_full['M_r_err'] = np.ones(n_galaxies)*0.05
    tab_full['g-i'] = np.zeros(n_galaxies)
    tab_full['g-i_err'] = np.ones(n_galaxies)*0.05/np.sqrt(2)
    
    res = np.append(tab_sample['r_e'][ww],tab_sample_cluster['Reff'][vv])
    redshifts = np.append(tab_sample['z_tonry'][ww],tab_sample_cluster['Z'][vv])
    M_rs_clust = [a - cluster_dist_mod(b) for a,b in zip(tab_sample_cluster['r_col'][vv],tab_sample_cluster['CATAID'][vv])]
    M_rs = np.append(tab_sample['M_r'][ww],M_rs_clust)
    g_is_clust = [a - b for a,b in zip(tab_sample_cluster['g_col'][vv],tab_sample_cluster['i_col'][vv])]
    g_is = np.append(tab_sample['g_i'][ww],g_is_clust)
    
    for i,catid in enumerate(tab_full['CATID']):
    
        ww = np.where(tab_sersic['CATAID'] == catid)[0]
        vr = np.where(tab_sersic_cluster_r['cataid'] == catid)[0]
        vg = np.where(tab_sersic_cluster_g['cataid_1'] == catid)[0]
        vi = np.where(tab_sersic_cluster_i['cataid_1'] == catid)[0]
        redshift = redshifts[i]
        re = res[i]
        re = re_arcsecs_to_re_kpc(re,redshift)
        M_r = M_rs[i]# - kcorrect_mag('r',g_rs[i],redshift)
        g_i = g_is[i] - kcorrect_mag('g',g_is[i],redshift) + kcorrect_mag('i',g_is[i],redshift)
        
        tab_full['M_r'][i] = M_r
        tab_full['g-i'][i] = g_i
        tab_full['Re'][i] = re
        
        if len(ww) > 0:
        
            if (tab_sersic['R_NCOMP'][ww[0]] == 1.5) or (tab_sersic['R_NCOMP'][ww[0]] == 2):
                disk_mag = tab_sersic['R_D_D_MAG'][ww[0]]
                bulge_mag = tab_sersic['R_D_B_MAG'][ww[0]]
                bulge_to_total = 1.0/(1+ 10.0**(0.4*(bulge_mag - disk_mag)))
            
                disk_mag_err = tab_sersic['R_D_D_MAG_ERR'][ww[0]]
                bulge_mag_err = tab_sersic['R_D_B_MAG_ERR'][ww[0]]
                bt_err = ((0.4*np.log(10)*10.0**(0.4*(bulge_mag - disk_mag))/
                            (1+10.0**(0.4*(bulge_mag - disk_mag)))**2)*
                            np.sqrt(bulge_mag_err**2 + disk_mag_err**2))

                disk_re_g_arcsec = tab_sersic['G_D_D_RE'][ww[0]]
                disk_re_i_arcsec = tab_sersic['I_D_D_RE'][ww[0]]
                disk_re_g_kpc = re_arcsecs_to_re_kpc(disk_re_g_arcsec,redshift)
                disk_re_i_kpc = re_arcsecs_to_re_kpc(disk_re_i_arcsec,redshift)
                disk_h_g_kpc = re_to_scalelength(disk_re_g_kpc)
                disk_h_i_kpc = re_to_scalelength(disk_re_i_kpc)

                disk_re_g_arcsec_err = tab_sersic['G_D_D_RE_ERR'][ww[0]]
                disk_re_i_arcsec_err = tab_sersic['I_D_D_RE_ERR'][ww[0]]
                disk_re_g_kpc_err = re_arcsecs_to_re_kpc(disk_re_g_arcsec_err,redshift)
                disk_re_i_kpc_err = re_arcsecs_to_re_kpc(disk_re_i_arcsec_err,redshift)
                disk_h_g_kpc_err = re_to_scalelength(disk_re_g_kpc_err)
                disk_h_i_kpc_err = re_to_scalelength(disk_re_i_kpc_err)

                tab_full['B/T'][i] = bulge_to_total
                tab_full['B/T_err'][i] = bt_err
                tab_full['h_g'][i] = np.log10(disk_h_g_kpc)
                tab_full['h_i'][i] = np.log10(disk_h_i_kpc)
                tab_full['h_g_err'][i] = disk_h_g_kpc_err
                tab_full['h_i_err'][i] = disk_h_i_kpc_err
                
            elif (tab_sersic['R_NCOMP'][ww[0]] == 1.0) and (tab_sersic['R_S_NSER'][ww[0]] > 0.75) and (tab_sersic['R_S_NSER'][ww[0]] < 1.5) and inc_ss:
                disk_re_g_arcsec = tab_sersic['G_S_RE'][ww[0]]    
                disk_re_i_arcsec = tab_sersic['I_S_RE'][ww[0]]
                disk_re_g_kpc = re_arcsecs_to_re_kpc(disk_re_g_arcsec,redshift)
                disk_re_i_kpc = re_arcsecs_to_re_kpc(disk_re_i_arcsec,redshift)
                disk_h_g_kpc = re_to_scalelength(disk_re_g_kpc)
                disk_h_i_kpc = re_to_scalelength(disk_re_i_kpc)
                
                disk_re_g_arcsec_err = tab_sersic['G_S_RE_ERR'][ww[0]]
                disk_re_i_arcsec_err = tab_sersic['I_S_RE_ERR'][ww[0]]
                disk_re_g_kpc_err = re_arcsecs_to_re_kpc(disk_re_g_arcsec_err,redshift)
                disk_re_i_kpc_err = re_arcsecs_to_re_kpc(disk_re_i_arcsec_err,redshift)
                disk_h_g_kpc_err = re_to_scalelength(disk_re_g_kpc_err)
                disk_h_i_kpc_err = re_to_scalelength(disk_re_i_kpc_err)
                
                tab_full['B/T'][i] = 0.0
                tab_full['B/T_err'][i] = 0.05
                tab_full['h_g'][i] = np.log10(disk_h_g_kpc)
                tab_full['h_i'][i] = np.log10(disk_h_i_kpc)
                tab_full['h_g_err'][i] = disk_h_g_kpc_err
                tab_full['h_i_err'][i] = disk_h_i_kpc_err
                
            # Including pure bulges makes no sense, especially as they have no
            # disk scale length
                
            #elif (tab_sersic['R_NCOMP'][ww[0]] == 1.0) and (tab_sersic['R_S_NSER'][ww[0]] > 2) and inc_ss:
            #    disk_re_g_arcsec = tab_sersic['G_S_RE'][ww[0]]    
            #    disk_re_i_arcsec = tab_sersic['I_S_RE'][ww[0]]
            #    disk_re_g_kpc = re_arcsecs_to_re_kpc(disk_re_g_arcsec,redshift)
            #    disk_re_i_kpc = re_arcsecs_to_re_kpc(disk_re_i_arcsec,redshift)
            #    disk_h_g_kpc = re_to_scalelength(disk_re_g_kpc)
            #    disk_h_i_kpc = re_to_scalelength(disk_re_i_kpc)
                
            #    disk_re_g_arcsec_err = tab_sersic['G_S_RE_ERR'][ww[0]]
            #    disk_re_i_arcsec_err = tab_sersic['I_S_RE_ERR'][ww[0]]
            #    disk_re_g_kpc_err = re_arcsecs_to_re_kpc(disk_re_g_arcsec_err,redshift)
            #    disk_re_i_kpc_err = re_arcsecs_to_re_kpc(disk_re_i_arcsec_err,redshift)
            #    disk_h_g_kpc_err = re_to_scalelength(disk_re_g_kpc_err)
            #    disk_h_i_kpc_err = re_to_scalelength(disk_re_i_kpc_err)
                
            #    tab_full['B/T'][i] = 1.0
            #    tab_full['B/T_err'][i] = 0.05
            #    tab_full['h_g'][i] = np.log10(disk_h_g_kpc)
            #    tab_full['h_i'][i] = np.log10(disk_h_i_kpc)
            #    tab_full['h_g_err'][i] = disk_h_g_kpc_err
            #    tab_full['h_i_err'][i] = disk_h_i_kpc_err
                        
            else:
                tab_full['B/T'][i] = np.nan
                tab_full['B/T_err'][i] = np.nan
                tab_full['h_g'][i] = np.nan
                tab_full['h_i'][i] = np.nan
                tab_full['h_g_err'][i] = np.nan
                tab_full['h_i_err'][i] = np.nan
        
        elif (len(vr) > 0) and (len(vg) > 0) and (len(vi) > 0):
        
            if (tab_sersic_cluster_r['r_ncomp'][vr] == '2') or (tab_sersic_cluster_r['r_ncomp'][vr] == '2a'):
            
                disk_mag = tab_sersic_cluster_r['r_d_mag2'][vr[0]]
                bulge_mag = tab_sersic_cluster_r['r_d_mag1'][vr[0]]
                bulge_to_total = 1.0/(1+ 10.0**(0.4*(bulge_mag - disk_mag)))
            
                disk_mag_err = tab_sersic_cluster_r['r_d_mag2_err'][vr[0]]
                bulge_mag_err = tab_sersic_cluster_r['r_d_mag1_err'][vr[0]]
                bt_err = ((0.4*np.log(10)*10.0**(0.4*(bulge_mag - disk_mag))/
                            (1+10.0**(0.4*(bulge_mag - disk_mag)))**2)*
                            np.sqrt(bulge_mag_err**2 + disk_mag_err**2))
                        
                disk_re_g_arcsec = tab_sersic_cluster_g['g_d_re2'][vg[0]]
                disk_re_i_arcsec = tab_sersic_cluster_i['i_d_re2'][vi[0]] 
                disk_re_g_kpc = re_arcsecs_to_re_kpc(disk_re_g_arcsec,redshift)
                disk_re_i_kpc = re_arcsecs_to_re_kpc(disk_re_i_arcsec,redshift)
                disk_h_g_kpc = re_to_scalelength(disk_re_g_kpc)
                disk_h_i_kpc = re_to_scalelength(disk_re_i_kpc)

                disk_re_g_arcsec_err = tab_sersic_cluster_g['g_d_re2_err'][vg[0]]
                disk_re_i_arcsec_err = tab_sersic_cluster_i['i_d_re2_err'][vi[0]]                       
                disk_re_g_kpc_err = re_arcsecs_to_re_kpc(disk_re_g_arcsec_err,redshift)
                disk_re_i_kpc_err = re_arcsecs_to_re_kpc(disk_re_i_arcsec_err,redshift)                    
                disk_h_g_kpc_err = re_to_scalelength(disk_re_g_kpc_err)
                disk_h_i_kpc_err = re_to_scalelength(disk_re_i_kpc_err)

                tab_full['B/T'][i] = bulge_to_total
                tab_full['B/T_err'][i] = bt_err
                tab_full['h_g'][i] = np.log10(disk_h_g_kpc)
                tab_full['h_i'][i] = np.log10(disk_h_i_kpc)
                tab_full['h_g_err'][i] = disk_h_g_kpc_err
                tab_full['h_i_err'][i] = disk_h_i_kpc_err
                
            elif (tab_sersic_cluster_r['r_ncomp'][vr] == '1') and (tab_sersic_cluster_r['r_s_nser'][vr] > 0.75) and (tab_sersic_cluster_r['r_s_nser'][vr] < 1.5) and inc_ss:
                
                disk_re_g_arcsec = tab_sersic_cluster_g['g_s_re'][vg[0]]
                disk_re_i_arcsec = tab_sersic_cluster_i['i_s_re'][vi[0]] 
                disk_re_g_kpc = re_arcsecs_to_re_kpc(disk_re_g_arcsec,redshift)
                disk_re_i_kpc = re_arcsecs_to_re_kpc(disk_re_i_arcsec,redshift)
                disk_h_g_kpc = re_to_scalelength(disk_re_g_kpc)
                disk_h_i_kpc = re_to_scalelength(disk_re_i_kpc)

                disk_re_g_arcsec_err = tab_sersic_cluster_g['g_s_re_err'][vg[0]]
                disk_re_i_arcsec_err = tab_sersic_cluster_i['i_s_re_err'][vi[0]]                       
                disk_re_g_kpc_err = re_arcsecs_to_re_kpc(disk_re_g_arcsec_err,redshift)
                disk_re_i_kpc_err = re_arcsecs_to_re_kpc(disk_re_i_arcsec_err,redshift)                    
                disk_h_g_kpc_err = re_to_scalelength(disk_re_g_kpc_err)
                disk_h_i_kpc_err = re_to_scalelength(disk_re_i_kpc_err)                
                
                tab_full['B/T'][i] = 0.0
                tab_full['B/T_err'][i] = 0.05
                tab_full['h_g'][i] = np.log10(disk_h_g_kpc)
                tab_full['h_i'][i] = np.log10(disk_h_i_kpc)
                tab_full['h_g_err'][i] = disk_h_g_kpc_err
                tab_full['h_i_err'][i] = disk_h_i_kpc_err
                
            #elif (tab_sersic_cluster_r['r_ncomp'][vr] == '1') and (tab_sersic_cluster_r['r_s_nser'][vr] > 2.5) and inc_ss:
            #    
            #    disk_re_g_arcsec = tab_sersic_cluster_g['g_s_re'][vg[0]]
            #    disk_re_i_arcsec = tab_sersic_cluster_i['i_s_re'][vi[0]] 
            #    disk_re_g_kpc = re_arcsecs_to_re_kpc(disk_re_g_arcsec,redshift)
            #    disk_re_i_kpc = re_arcsecs_to_re_kpc(disk_re_i_arcsec,redshift)
            #    disk_h_g_kpc = re_to_scalelength(disk_re_g_kpc)
            #    disk_h_i_kpc = re_to_scalelength(disk_re_i_kpc)

            #    disk_re_g_arcsec_err = tab_sersic_cluster_g['g_s_re_err'][vg[0]]
            #    disk_re_i_arcsec_err = tab_sersic_cluster_i['i_s_re_err'][vi[0]]                       
            #    disk_re_g_kpc_err = re_arcsecs_to_re_kpc(disk_re_g_arcsec_err,redshift)
            #    disk_re_i_kpc_err = re_arcsecs_to_re_kpc(disk_re_i_arcsec_err,redshift)                    
            #    disk_h_g_kpc_err = re_to_scalelength(disk_re_g_kpc_err)
            #    disk_h_i_kpc_err = re_to_scalelength(disk_re_i_kpc_err)                
                
            #    tab_full['B/T'][i] = 1.0
            #    tab_full['B/T_err'][i] = 0.05
            #    tab_full['h_g'][i] = np.log10(disk_h_g_kpc)
            #    tab_full['h_i'][i] = np.log10(disk_h_i_kpc)
            #    tab_full['h_g_err'][i] = disk_h_g_kpc_err
            #    tab_full['h_i_err'][i] = disk_h_i_kpc_err
                
                
            else:
                tab_full['B/T'][i] = np.nan
                tab_full['B/T_err'][i] = np.nan
                tab_full['h_g'][i] = np.nan
                tab_full['h_i'][i] = np.nan
                tab_full['h_g_err'][i] = np.nan
                tab_full['h_i_err'][i] = np.nan
        #ww = np.where(tab_ap['CATID'] == catid)[0]
        
        #if len(ww) > 0:
        #    tab_full['M_r_err'][i] = tab_ap['MAGERR_AUTO_R'][ww]*3
        #    tab_full['g-i_err'][i] = np.sqrt(tab_ap['MAGERR_AUTO_G'][ww]**2 + tab_ap['MAGERR_AUTO_I'][ww]**2)
            
                    
    return tab_full
    
def clean_galaxy_data(tab_galaxy):

    ww = np.where((tab_galaxy['h_g'] > 0.0) & (tab_galaxy['h_i'] > 0.0))

    return tab_galaxy[ww]
    
def extinction_corrected_gr(tab_sample,tab_ap,tab_ex):

    # Calculate the extinction-corrected g-r colour from GAMA data

    g_rs = []
    
    for catid in tab_sample['CATID']:
        ww1 = np.where(tab_ap['CATID'] == catid)[0]
        ww2 = np.where(tab_ex['CATID'] == catid)[0]

        gmag = tab_ap['MAG_AUTO_G'][ww1].data
        rmag = tab_ap['MAG_AUTO_R'][ww1].data
    
        gex = 0.0#tab_ex['A_g_1'][ww2].data
        rex = 0.0#tab_ex['A_r'][ww2].data
    
        g_r = (gmag - gex) - (rmag - rex)
        g_rs.append(g_r)
    
    return g_rs
    
    
def kcorrect_mag(band,colour,z_spec):

    # k-correct an input magnitude to redshift 0 - possibly apply this to colours too?

    # mr = Mq + DM + Kqr
    
    # mr = apparent mag, Mq = emitted frame absolute magnitude, DM = distance modulus, Kqr = k-correction

    if band == 'r':
        kcorr = calc_kcor.calc_kcor(band,z_spec,'g - r',colour)
    else:
        kcorr = calc_kcor.calc_kcor(band,z_spec,'g - i',colour)
    
    return kcorr

    
def re_arcsecs_to_re_kpc(rad_arcsec,redshift):

    # Convert angular measures to physical measures assuming the SAMI cosmology
    
    cosmo = FlatLambdaCDM(H0=70,Om0=0.3)
    arcmin_to_kpc = cosmo.kpc_proper_per_arcmin(redshift)
    rad_arcminute = (rad_arcsec*u.arcsec).to(u.arcminute)
    rad_kpc = rad_arcminute*arcmin_to_kpc

    return rad_kpc.value
    
def re_to_scalelength(re):

    # From Graham & Driver (2005), equation 15 + 16
    
    h = re/1.678

    return h

def rdisk_BVR_to_ugriz(mw_dat):

    # Interpolate the MW disk scale lengths from BVR to ugriz,
    # assuming scale length is a monotonic, smoothly changing function
    
    band_centres ={'u':352,'g':480,'r':625,'i':769,'z':911,
        'U':365,'B':445,'V':551,'R':658,'I':806,'K':2190}
        
    def fit_func(x,a,b,c,d,e,f):
        return a*(x**5) + b*(x**4) + c*(x**3) + d*(x**2) + e*x + f
         
    x = [band_centres['B'],band_centres['V'],band_centres['R'],band_centres['K'],
        1000,1250,1500,1750,2000]
    y = [mw_dat['h_B'][0],mw_dat['h_V'][0],mw_dat['h_R'][0],mw_dat['h_K'][0],
        2.5,2.5,2.5,2.5,2.5]
    yerr = [mw_dat['h_B'][1],mw_dat['h_V'][1],mw_dat['h_R'][1],mw_dat['h_K'][1],
        0.25,0.25,0.25,0.25,0.25]
    popt,pcov = curve_fit(fit_func,x,y,sigma=yerr)
    popt2,pcov2 = curve_fit(fit_func,x,yerr)
    
    mw_dat['h_u'] = [fit_func(band_centres['u'],*popt),fit_func(band_centres['u'],*popt2)]
    mw_dat['h_g'] = [fit_func(band_centres['g'],*popt),fit_func(band_centres['g'],*popt2)]
    mw_dat['h_r'] = [fit_func(band_centres['r'],*popt),fit_func(band_centres['r'],*popt2)]
    mw_dat['h_i'] = [fit_func(band_centres['i'],*popt),fit_func(band_centres['i'],*popt2)]
    mw_dat['h_z'] = [fit_func(band_centres['z'],*popt),fit_func(band_centres['z'],*popt2)]
    
    return mw_dat
    
def load_mw_data():

    # Read Milky Way properties from a file and return
    
    tab = Table.read('/Users/nscott/Data/MW_analogues/mw_data.dat',
        format='ascii.commented_header')
    
    # Reformat into a dictionary
    
    mw_dict = {}
    
    for i in range(len(tab)):
        mw_dict[tab['Quantity'][i]] = [tab['Value'][i],tab['Error'][i]]
        
    mw_dict['g-i'] = [mw_dict['g-r'][0] + mw_dict['r-i'][0],np.sqrt(mw_dict['g-r'][1]**2+mw_dict['r-i'][1]**2)]
        
    return mw_dict

def calc_dist(pos_gal,pos_mw,err_gal,err_mw,n_iter=10000):

    # Calculate the euclidean distance between two n-dimensional points
    # accounting for the uncertainties by treating them as Gaussian PDFs
    # and using pair-wise sampling

    dists = []
    for i in np.arange(n_iter):
        pos_gal_new = [pos + (np.random.randn()*sig) for pos,sig in zip(pos_gal,err_gal)]
        pos_mw_new = [pos + (np.random.randn()*sig) for pos,sig in zip(pos_mw,err_mw)]
        dist = distance.euclidean(pos_gal_new,pos_mw_new)
            
        dists.append(dist)
        
    return np.mean(dists)
    
def normalize(data):

    # Return a normalised data array that has had the mean subtracted and
    # been normalised to unit standard deviation
    mn = np.nanmedian(data)
    sig = np.nanstd(data)
    
    data_new = (data-mn)/sig
    
    return data_new,mn,sig

def euclidean_error_comparison(dim=5):

    # Test whether the minimum distance between two points is affected
    # by their random uncertainties
    
    pos1 = np.random.random(dim)*50
    
    pos2 = np.random.random(dim)*50
    
    dist_no_error = distance.euclidean(pos1,pos2)
    
    dists_small_sig = []
    dists_big_sig = []
    
    sig1 = np.random.random(dim)
    sig2 = np.random.random(dim)
    
    print('Pos1: ',pos1)
    print('Pos2: ',pos2)
    
    print('Sig1: ',sig1)
    print('Sig2: ',sig2)
    
    for i in np.arange(100000):
        pos1_new = [pos+(np.random.randn()*sig*0.5) for pos,sig in zip(pos1,sig1)]
        pos2_new = [pos+(np.random.randn()*sig*0.5)for pos,sig in zip(pos2,sig2)]
        dist = distance.euclidean(pos1_new,pos2_new)
        dists_small_sig.append(dist)

        pos1_new = [pos+(np.random.randn()*sig*25) for pos,sig in zip(pos1,sig1)]
        pos2_new = [pos+(np.random.randn()*sig*25) for pos,sig in zip(pos2,sig2)]
        dist = distance.euclidean(pos1_new,pos2_new)
        dists_big_sig.append(dist)
        
    dist_error_small = np.mean(dists_small_sig)
    dist_error_big = np.mean(dists_big_sig)
    
    print()
    print('Euclidean distance ignoring uncertainties:')
    print(dist_no_error)
    print()
    print('Euclidean distance including small uncertainties:')
    print(dist_error_small)
    print()
    print('Euclidean distance including large uncertainties:')
    print(dist_error_big)
    
    # Conclusion is that it is, when the errors are significant. Need to use
    # pairwise sampling, can't just take the expected values
