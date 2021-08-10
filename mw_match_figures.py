from astropy.table import Table
import numpy as np
import mw_match
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import glob

def analogue_topx_selection_hists(x,inc_ss=True):

    data = load_data(inc_ss=inc_ss)
    mw_data = mw_match.load_mw_data()
    mw_data = mw_match.rdisk_BVR_to_ugriz(mw_data)

    ind = np.argsort(data['dist'])[:x]
    
    fig = plt.figure(figsize=(15,8)) 

    fig.add_subplot(2,3,1)
    analogue_hist(data['Mstar'],ind,label=r'M$_\star$',val=mw_data['Mstar'][0])

    fig.add_subplot(2,3,2)
    analogue_hist(data['g-i'],ind,label='(g-i)',val=mw_data['g-i'][0])
        
    fig.add_subplot(2,3,3)
    analogue_hist(data['B/T'],ind,label='B/T',val=mw_data['B/T'][0])
        
    fig.add_subplot(2,3,4)
    analogue_hist(data['h_g'],ind,label=r'log h$_g$',val=np.log10(mw_data['h_g'][0]))
    
    fig.add_subplot(2,3,5)
    analogue_hist(data['h_i'],ind,label=r'log h$_i$',val=np.log10(mw_data['h_i'][0]))  
        
    plt.tight_layout()

def analogue_topx_props_hists(x,inc_ss=True):

    data = load_data(inc_ss=inc_ss)
    kinfile = '/Users/nscott/Data/SAMI/Survey/jvds_stelkin_cat_v012_mge_seecorr_kh20_v260421_private.fits'
    kintab = Table.read(kinfile)
    sspfile = '/Users/nscott/Data/SAMI/Survey/Stellar Populations/SSPAperturesDR3.fits'
    ssptab = Table.read(sspfile)
    
    lr,t,z,alpha = [],[],[],[]
    for id in data['CATID']:
        ww = np.where(kintab['CATID_EXT'] == int(id))[0]
        if len(ww) > 0:
            lr.append(kintab['LAMBDAR_RE_EO'][ww])
        else:
            lr.append(np.nan)
        ww = np.where(ssptab['CATIDPUB'] == str(int(id))+'_A')[0]
        if len(ww) > 0:
            t.append(ssptab['Age_RE_MGE'][ww].data[0])
            z.append(ssptab['Z_RE_MGE'][ww].data[0])
            alpha.append(ssptab['alpha_RE_MGE'][ww].data[0])
        else:
            t.append(np.nan)
            z.append(np.nan)
            alpha.append(np.nan)
    
    lr,t,z,alpha = np.asarray(lr),np.asarray(t),np.asarray(z),np.asarray(alpha)
    z[z<-10] = np.nan
    alpha[alpha<-10] = np.nan 
    t[t<-10] = np.nan   

    ind = np.argsort(data['dist'])[:x]
    
    fig = plt.figure(figsize=(10,8)) 

    fig.add_subplot(2,2,1)
    analogue_hist(lr,ind,label=r'$\lambda_{R,0}$')

    fig.add_subplot(2,2,2)
    analogue_hist(np.log10(t),ind,label='log Age')
        
    fig.add_subplot(2,2,3)
    analogue_hist(z,ind,label='[Z/H]')
        
    fig.add_subplot(2,2,4)
    analogue_hist(alpha,ind,label=r'[$\alpha$/Fe]')  
        
    plt.tight_layout()

def analogue_hist(x,ind,val=None,label=''):

    h = plt.hist(x,normed=True,bins=20 )
    plt.hist(x[ind],normed=True,alpha=0.5,bins=h[1])
    plt.ylabel('Frequency',fontsize=15)
    plt.xlabel(label,fontsize=15)
    if val is None:
        val = np.nanmedian(x[ind])
    plt.autoscale(False)
    plt.plot([val,val],[0,100],'k--')
    yloc = plt.ylim()[1]*.9
    xloc = plt.xlim()[0]+0.05*(plt.xlim()[1]-plt.xlim()[0])
    plt.text(xloc,yloc,"{0:.3g}".format(val),fontsize=15)


def analogue_selection_plots(inc_ss=True):

    data = load_data(inc_ss=inc_ss)
    mw_data = mw_match.load_mw_data()
    mw_data = mw_match.rdisk_BVR_to_ugriz(mw_data)
    
    fig = plt.figure(figsize=(10,8))
    
    fig.add_subplot(2,2,1)
    analogue_xydist(data['Mstar'],data['g-i'],data['dist'],
        labels=[r'log M$_{\star}$','(g-i)','log Similarity Distance'],
        mw=[mw_data['Mstar'][0],mw_data['g-i'][0]])

    fig.add_subplot(2,2,2)
    analogue_xydist(data['Mstar'],data['B/T'],data['dist'],
        labels=[r'log M$_{\star}$','B/T','log Similarity Distance'],
        mw=[mw_data['Mstar'][0],mw_data['B/T'][0]])
        
    fig.add_subplot(2,2,3)
    analogue_xydist(data['Mstar'],data['h_g'],data['dist'],
        labels=[r'log M$_{\star}$',r'log h$_g$','log Similarity Distance'],
        mw=[mw_data['Mstar'][0],np.log10(mw_data['h_g'][0])])
        
    fig.add_subplot(2,2,4)
    analogue_xydist(data['Mstar'],data['h_i'],data['dist'],
        labels=[r'log M$_{\star}$',r'log h$_i$','log Similarity Distance'],
        mw=[mw_data['Mstar'][0],np.log10(mw_data['h_i'][0])])  
        
    plt.tight_layout() 
    
def analogue_lreps(inc_ss=True):     
  
    data = load_data(inc_ss=inc_ss)
    kinfile = '/Users/nscott/Data/SAMI/Survey/jvds_stelkin_cat_v012_mge_seecorr_kh20_v260421_private.fits'
    kintab = Table.read(kinfile)
    
    fig = plt.figure()
    
    
    eps,lr = [],[]
    for id in data['CATID']:
        ww = np.where(kintab['CATID_EXT'] == int(id))[0]
        if len(ww) > 0:
            eps.append(kintab['ELLIP'][ww])
            lr.append(kintab['LAMBDAR_RE'][ww])
        else:
            eps.append(np.nan)
            lr.append(np.nan)
        
    analogue_xydist(eps,lr,data['dist'],
        labels=[r'$\epsilon_e$',r'$\lambda_R$','log Similarity Distance'])
    plt.tight_layout()
    

def analogue_propmass(inc_ss=True):

    data = load_data(inc_ss=inc_ss)
    kinfile = '/Users/nscott/Data/SAMI/Survey/jvds_stelkin_cat_v012_mge_seecorr_kh20_v260421_private.fits'
    kintab = Table.read(kinfile)
    sspfile = '/Users/nscott/Data/SAMI/Survey/Stellar Populations/SSPAperturesDR3.fits'
    ssptab = Table.read(sspfile)
    
    lr,t,z,alpha = [],[],[],[]
    for id in data['CATID']:
        ww = np.where(kintab['CATID_EXT'] == int(id))[0]
        if len(ww) > 0:
            lr.append(kintab['LAMBDAR_RE_EO'][ww])
        else:
            lr.append(np.nan)
        ww = np.where(ssptab['CATIDPUB'] == str(int(id))+'_A')[0]
        if len(ww) > 0:
            t.append(ssptab['Age_RE_MGE'][ww].data[0])
            z.append(ssptab['Z_RE_MGE'][ww].data[0])
            alpha.append(ssptab['alpha_RE_MGE'][ww].data[0])
        else:
            t.append(np.nan)
            z.append(np.nan)
            alpha.append(np.nan)
    
    fig = plt.figure(figsize=(10,8)) 
    

    t,z,alpha = np.asarray(t),np.asarray(z),np.asarray(alpha)
    z[z<-10] = np.nan
    alpha[alpha<-10] = np.nan
        
    fig.add_subplot(2,2,1)
    analogue_xydist(data['Mstar'],lr,data['dist'],topx=10,
        labels=[r'log M$_{\star}$',r'$\lambda_{R,0}$','log Similarity Distance'])  
    fig.add_subplot(2,2,2)
    vv = np.isfinite(np.log10(t))
    analogue_xydist(data['Mstar'][vv],np.log10(t[vv]),data['dist'][vv],topx=10,
        labels=[r'log M$_{\star}$','log Age','log Similarity Distance'])    
    fig.add_subplot(2,2,3)
    vv = np.isfinite(z)
    analogue_xydist(data['Mstar'][vv],z[vv],data['dist'][vv],topx=10,
        labels=[r'log M$_{\star}$','[Z/H]]','log Similarity Distance'])
    fig.add_subplot(2,2,4)
    np.isfinite(alpha)
    analogue_xydist(data['Mstar'][vv],alpha[vv],data['dist'][vv],topx=10,
        labels=[r'log M$_{\star}$',r'[$\alpha$/Fe]','log Similarity Distance']) 
    plt.tight_layout()     

def analogue_xydist(xs,ys,dist,mw=None,topx=False,labels=['','',''],log=True):

    if log:
        dist = np.log10(dist)
    cnorm = colors.Normalize(vmin=np.min(logdist),vmax=np.max(dist))
    scalarmap = cm.ScalarMappable(norm=cnorm,cmap=plt.get_cmap('YlOrRd_r'))
    
    for x,y,d in zip(xs,ys,dist):
        plt.plot(x,y,'o',color=scalarmap.to_rgba(d,alpha=0.5))
    plt.xlabel(labels[0],fontsize=15)
    plt.ylabel(labels[1],fontsize=15)
    
    if topx != False:
        ww = np.argsort(dist)[:topx]
        for x,y in zip(np.asarray(xs)[ww],np.asarray(ys)[ww]):
            plt.plot(x,y,'kx',ms=5)
    
    if mw != None:
        plt.plot(mw[0],mw[1],'ks',ms=10)
    
    scalarmap.set_array(dist)
    cbar = plt.colorbar(scalarmap)
    cbar.set_label(labels[2],fontsize=15)
    plt.draw()
    plt.show()


def analogue_size_mag_selections():

    tables = glob.glob('/Users/nscott/Data/MW_analogues/SAMI_analogues/tables/*/sami_analogues_ss.dat')
    tables.append('/Users/nscott/Data/MW_analogues/SAMI_analogues/tables/sami_analogues_ss.dat')
    tables[:] = [tables[i] for i in [6,0,1,2,3,4,5]]
    
    data0 = load_data()

    titles = ['All']
    for table_file in tables[1:]:
        tmp = table_file.split('/')[7]
        titles.append(tmp)

    fig = plt.figure(figsize=(21,6))

    for i,table_file in enumerate(tables):
        fig.add_subplot(2,4,i+1)
        data = Table.read(table_file,format='ascii')
        analogue_xydist(data0['M_r'],np.log10(data0['Re']),data['Distance']/np.std(data['Distance']),
            labels=[r'M$_r$',r'log R$_e$','log Similarity Distance'])
        plt.title(titles[i])
        plt.axis([-15,-24,-0.25,1.3])
    plt.tight_layout()
        

    
def load_data(inc_ss=True):

    gd = mw_match.load_galaxy_data(inc_ss=inc_ss)
    gd = mw_match.clean_galaxy_data(gd)
    sample_file = '/Users/nscott/Data/MW_analogues/SAMI_analogues/tables/sami_analogues_ss.dat'
    tb = Table.read(sample_file,format='ascii')
    
    match = []
    for id in gd['CATID']:
        ww = np.where(int(id) == tb['CATID'])[0]
        try:
            match.append(ww[0])
        except:
            print(id)
            raise Exception()
    
    gd['dist'] = tb['Distance'][match]
    
    return gd