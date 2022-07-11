spei=nc_varget('spei01.nc','spei'); %1 month
for tm=[1:1379]
    spei(tm,:,:)=flipud(squeeze(spei(tm,:,:))); 
end
save('spei-cl.mat','spei','-v7.3');
%% image reshape, remove anomalism points in precipitation
for yr=[1901:2012]
    tas_mx=[];
    %prcp_mx=[];
    tp=nc_varget(['/Volumes/My Passport/LAB-NAU/drought/data-temp/tas_daily_',num2str(yr),'-',num2str(yr),'.nc'],'tas');
    tp_ori=tp;
    tp(:,:,[1:360])=tp(:,:,[361:720]);
    tp(:,:,[361:720])=tp_ori(:,:,[1:360]);
    %pc=nc_varget(['/Volumes/My Passport/LAB-NAU/drought/data-prcp/prcp_daily_',num2str(yr),'-',num2str(yr),'.nc'],'prcp');
    %pc_ori=pc;
    %pc(:,:,[1:360])=pc(:,:,[361:720]);
    %pc(:,:,[361:720])=pc_ori(:,:,[1:360]);
    %[m,n,r]=size(pc);
    %pc_thr=prctile(reshape(pc,1,m*360*720),99);
    %pc(pc>pc_thr)=pc_thr;
    [m,n,r]=size(tp);
    for tm=[1:m]
        tp(tm,:,:)=flipud(squeeze(tp(tm,:,:)));
        %pc(tm,:,:)=flipud(squeeze(pc(tm,:,:)));
    end
    save(['/Volumes/My Passport/LAB-NAU/drought/data-temp-cl/tas-',num2str(yr),'.mat'],'tp')
    %save(['/Volumes/My Passport/LAB-NAU/drought/data-prcp-cl/prcp-',num2str(yr),'.mat'],'pc')
end 

%% transfer daily T,P to monthly T,P in 2000~2003
mn=[31,28,31,30,31,30,31,31,30,31,30,30,31];
monTall=[];
monPall=[];
for yr=[2000:2003]
    load(['/Volumes/My Passport/LAB-NAU/drought/data-temp-cl/tas-',num2str(yr),'.mat']);
    load(['/Volumes/My Passport/LAB-NAU/drought/data-prcp-cl/prcp-',num2str(yr),'.mat']);
    if(~mod(yr,4))
        mn(2)=29;
    else
        mn(2)=28;
    end
    for i=[1:12]
        s=sum(mn(1:i));
        monT=sum(tp([(s-mn(i)+1):s],:,:),1)/mn(i);
        monTall=[monTall;monT];
        monP=sum(pc([(s-mn(i)+1):s],:,:),1);
        monPall=[monPall;monP];
    end
end
save('/Volumes/My Passport/LAB-NAU/drought/data-temp-cl/monTemp.mat','monTall');
save('/Volumes/My Passport/LAB-NAU/drought/data-prcp-cl/monPrcp.mat','monPall');

%% multivariate regression
load('/Volumes/My Passport/LAB-NAU/drought/data-temp-cl/monTemp.mat');
load('/Volumes/My Passport/LAB-NAU/drought/data-prcp-cl/monPrcp.mat');
load('spei-cl.mat');
Y=spei([((2000-1901)*12+1):(2004-1901)*12],:,:);
[m,n]=find(~isnan(squeeze(Y(1,:,:))));
Bmap=zeros(3,360,720);
Rmap=zeros(360,720);
Pmap=zeros(360,720);
for i=[1:size(m,1)]
    lat=m(i);
    lon=n(i);
    T=reshape(monTall(:,lat,lon),48,1);
    P=reshape(monPall(:,lat,lon),48,1);
    [b, bint,r,rint,stats]=regress(Y(:,lat,lon),[ones(48,1) 0.001*T 1000*P]);
    Bmap(:,lat,lon)=b;
    Rmap(lat,lon)=stats(1);
    Pmap(lat,lon)=stats(3);
end
save('regression/Bmap.mat','Bmap');
save('regression/Rmap.mat','Rmap');
save('regression/Pmap.mat','Pmap');

%% single variate regression
load('/Volumes/My Passport/LAB-NAU/drought/data-temp-cl/monTemp.mat');
load('/Volumes/My Passport/LAB-NAU/drought/data-prcp-cl/monPrcp.mat');
load('spei-cl.mat');
Y=spei([((2000-1901)*12+1):(2004-1901)*12],:,:);
[m,n]=find(~isnan(squeeze(Y(1,:,:))));
Bmap1=zeros(2,360,720);
Rmap1=zeros(360,720);
Pmap1=zeros(360,720);
for i=[1:size(m,1)]
    lat=m(i);
    lon=n(i);
    T=reshape(monTall(:,lat,lon),48,1);
    
    [b, bint,r,rint,stats]=regress(Y(:,lat,lon),[ones(48,1) 0.001*T]);
    Bmap1(:,lat,lon)=b;
    Rmap1(lat,lon)=stats(1);
    Pmap1(lat,lon)=stats(3);
end
save('regression/Bmap1.mat','Bmap1');
save('regression/Rmap1.txt','Rmap1','-ascii');
save('regression/Pmap1.txt','Pmap1','-ascii');
        
%% generate daily SPEI according to the regression b items
for yr=[1981:2012]
    tas_mx=[];
    prcp_mx=[];
    tp=nc_varget(['/Volumes/My Passport/LAB-NAU/drought/data-temp/tas_daily_',num2str(yr),'-',num2str(yr),'.nc'],'tas');
    tp_ori=tp;
    tp(:,:,[1:360])=tp(:,:,[361:720]);
    tp(:,:,[361:720])=tp_ori(:,:,[1:360]);
    pc=nc_varget(['/Volumes/My Passport/LAB-NAU/drought/data-prcp/prcp_daily_',num2str(yr),'-',num2str(yr),'.nc'],'prcp');
    pc_ori=pc;
    pc(:,:,[1:360])=pc(:,:,[361:720]);
    pc(:,:,[361:720])=pc_ori(:,:,[1:360]);
    [m,n,r]=size(pc);
    pc_thr=prctile(reshape(pc,1,m*360*720),99);
    pc(pc>pc_thr)=pc_thr;
    for tm=[1:m]
        tp(tm,:,:)=flipud(squeeze(tp(tm,:,:)));
        pc(tm,:,:)=flipud(squeeze(pc(tm,:,:)));
    end
    save(['/Volumes/My Passport/LAB-NAU/drought/data-temp-cl/tas-',num2str(yr),'.mat'],'tp')
    save(['/Volumes/My Passport/LAB-NAU/drought/data-prcp-cl/prcp-',num2str(yr),'.mat'],'pc')
end 

load('regression/Bmap.mat');
load('spei-cl.mat');
slist=[];
for yr=[1910:2012]
    %load(['/Volumes/My Passport/LAB-NAU/drought/data-temp-cl/tas-',num2str(yr),'.mat']);
   
    tp=nc_varget(['/Volumes/My Passport/LAB-NAU/drought/data-temp/tas_daily_',num2str(yr),'-',num2str(yr),'.nc'],'tas');
    tp_ori=tp;
    tp(:,:,[1:360])=tp(:,:,[361:720]);
    tp(:,:,[361:720])=tp_ori(:,:,[1:360]);
    [m,n,r]=size(tp);
    for tm=[1:m]
        tp(tm,:,:)=flipud(squeeze(tp(tm,:,:)));
    end
    save(['/Volumes/My Passport/LAB-NAU/drought/data-temp-cl/tas-',num2str(yr),'.mat'],'tp')
    load(['/Volumes/My Passport/LAB-NAU/drought/data-prcp-cl/prcp-',num2str(yr),'.mat']);
    [m,n,r]=size(pc);
    spei_reg=zeros(m,n,r);
    for dy=[1:m]
        spei_reg(dy,:,:)=Bmap(1,:,:)+0.001*Bmap(2,:,:).*tp(dy,:,:)+1000*Bmap(3,:,:).*pc(dy,:,:);
        s=spei_reg(dy,200,250);
        slist=[slist;s];
    end
    spei_reg(:,find(isnan(squeeze(spei(1,:,:)))))=NaN;
    spei_reg(spei_reg>3)=3;
    spei_reg(spei_reg<-3)=-3;
    for dy=[1:m]
        spei_reg_tr=mat2gray(squeeze(spei_reg(dy,:,:)));
        imshow(spei_reg_tr);
        saveas(gcf,['/Volumes/My Passport/LAB-NAU/drought/dataAll/',num2str(yr),'-',num2str(dy),'.png']);
    end
    save(['/Volumes/My Passport/LAB-NAU/drought/regSPEI/d-',num2str(yr),'.mat'],'spei_reg');
end
    
%% generate daily SPEI based on daily T
slist=[];
mn=[31,28,31,30,31,30,31,31,30,31,30,30,31];
load('spei-cl.mat');
for yr=[1901:2012]
    load(['/Volumes/My Passport/LAB-NAU/drought/data-temp-cl/tas-',num2str(yr),'.mat']);
    [m,n,r]=size(pc);
    speiT=zeros(m,n,r);
    if(~mod(yr,4))
        mn(2)=29;
    else
        mn(2)=28;
    end
    Y=spei([(yr-1901)*12+1,(yr-1901+1)*12],:,:);
    for i=[1:12]
        s=sum(mn(1:i));
        Y=spei();
        for lat=[1:360]
            for lon=[1:720]
                tday=tp([(s-mn(i)+1):s],lat,lon);
                tday_t=mat2gray(tday);
                avgTday_t=sum(tday_t)/mn(i);
                speiT([(s-mn(i)+1):s],lat,lon)=Y(i,lat,lon).*tday_t/avgTday_t;
            end
        end
    end
    
    save(['TpSPEI/d-',num2str(yr),'.mat'],'spei');
end
    
    