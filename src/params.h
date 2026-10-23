// params.h: params.txt reader with the defaults, option names and post-processing of XBeach r6167
// params.F90 (all_input) for wavemodel = nonh.
#pragma once
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>

static std::string lower(std::string s) { for (auto &ch : s) ch = (char)tolower((unsigned char)ch); return s; }
static std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

// readkey.F90: every line with '=' is a key; keys are case insensitive; the first occurrence wins.
// Everything after '%' is treated as a comment (XBeach itself would fail to parse such a value).
struct ParFile {
    std::vector<std::string> lines;
    std::vector<std::pair<std::string, std::string>> kv;
    std::set<std::string> used;
    explicit ParFile(const char *fn) {
        std::ifstream f(fn);
        if (!f) fail(std::string("cannot open ") + fn);
        std::string line;
        while (std::getline(f, line)) {
            for (auto &ch : line) if ((unsigned char)ch < 32 || (unsigned char)ch > 126) ch = ' ';
            lines.push_back(line);
            size_t pc = line.find('%'); std::string l = pc == std::string::npos ? line : line.substr(0, pc);
            size_t eq = l.find('=');
            if (eq == std::string::npos) continue;
            kv.push_back({lower(trim(l.substr(0, eq))), trim(l.substr(eq + 1))});
        }
    }
    const std::string *find(const std::string &key) {
        std::string k = lower(key);
        for (auto &p : kv) if (p.first == k) { used.insert(k); return &p.second; }
        return nullptr;
    }
    bool has(const std::string &key) { return find(key) != nullptr && !find(key)->empty(); }
    // readkey_dbl: read(value,'(f10.0)'): the first 10 characters, blanks ignored
    double dbl(const std::string &key, double def) {
        auto v = find(key);
        if (!v || v->empty()) return def;
        std::string s = v->substr(0, 10), t;
        for (char ch : s) if (ch != ' ' && ch != '\t') t.push_back(ch);
        for (auto &ch : t) if (ch == 'd' || ch == 'D') ch = 'e';
        char *e; double x = strtod(t.c_str(), &e);
        if (e == t.c_str()) fail("value of " + key + " cannot be interpreted: " + *v);
        return x;
    }
    int integer(const std::string &key, int def) {
        auto v = find(key);
        if (!v || v->empty()) return def;
        std::istringstream is(*v); long x;
        if (!(is >> x)) fail("value of " + key + " cannot be interpreted: " + *v);
        return (int)x;
    }
    std::string name(const std::string &key, const std::string &def = "") {
        auto v = find(key);
        if (!v || v->empty()) return def;
        std::istringstream is(*v); std::string w; is >> w;
        return w;
    }
    // readkey_dblvec: n list-directed values after "key ="; the default for all of them when the key is missing
    std::vector<double> dblvec(const std::string &key, int n, double def, const std::string &fn = "params.txt") {
        std::vector<double> out(n, def);
        auto v = find(key);
        if (!v || v->empty()) return out;
        std::string s = *v;
        for (auto &ch : s) if (ch == ',' || ch == '\t') ch = ' ';
        std::istringstream is(s);
        for (int q = 0; q < n; q++) {
            std::string w;
            if (!(is >> w)) fail(fn + ": " + key + " needs " + std::to_string(n) + " values");
            for (auto &ch : w) if (ch == 'd' || ch == 'D') ch = 'e';
            char *e; out[q] = strtod(w.c_str(), &e);
            if (e == w.c_str()) fail(fn + ": cannot read " + key);
        }
        return out;
    }
    // parmapply with setallowednames/setoldnames: the value is one of the names or its old numeric code
    int option(const std::string &key, const std::vector<std::string> &names, int defidx,
               const std::vector<std::string> &oldnames = {}) {
        std::string v = lower(name(key, names[defidx]));
        for (size_t q = 0; q < names.size(); q++) if (v == names[q]) return (int)q;
        for (size_t q = 0; q < oldnames.size(); q++) if (v == oldnames[q]) return (int)q;
        std::string msg = "invalid value '" + v + "' for " + key + " (allowed:";
        for (auto &n : names) msg += " " + n;
        fail(msg + ")");
        return -1;
    }
    // lines following "key = n" (readOutputStrings / readPointPosition)
    std::vector<std::string> list_after(const std::string &key, int n) {
        std::vector<std::string> out;
        size_t q = 0;
        for (; q < lines.size(); q++) {
            size_t eq = lines[q].find('=');
            if (eq != std::string::npos && lower(trim(lines[q].substr(0, eq))) == lower(key)) break;
        }
        for (q++; q < lines.size() && (int)out.size() < n; q++) out.push_back(trim(lines[q]));
        if ((int)out.size() < n) fail("params.txt: fewer than " + std::to_string(n) + " lines after " + key);
        return out;
    }
};

enum { WM_STAT = 0, WM_SURFBEAT = 1, WM_NONH = 2 };
// wbctype names in the order of params.F90 setallowednames
enum { WBC_PARAMS = 0, WBC_PARAMETRIC, WBC_SWAN, WBC_VARDENS, WBC_OFF, WBC_JONSTABLE, WBC_REUSE, WBC_TS1, WBC_TS2,
       WBC_TSNONH };
enum { OUT_FORTRAN = 0, OUT_NETCDF = 1, OUT_DEBUG = 2 };

struct Params {
    // processes
    int wavemodel = WM_NONH, cyclic = 0, swave = 0, lwave = 1, flow = 1, sedtrans = 0, morphology = 0,
        avalanching = 0, gwflow = 0, ships = 0, vegetation = 0, setbathy = 0, viscosity = 1, wind = 0,
        rainfall = 0, useXBeachG = 0;
    // grid
    int gridform = 0, nx = 0, ny = 0, vardx = 1;
    double xori = 0, yori = 0, alfa = 0, posdwn = 1, dx = -1, dy = -1;
    std::string depfile, xfile, yfile, xyfile;
    // time
    double CFL = 0.7, tstop = 2000, maxdtfac = 500, tstart = 0, tint = 900, tintg = 900, tintp = 1, tintm = -123,
           tinth = 0;
    int defuse = 1;
    std::string tsglobal, tspoints, tsmean, tshotstart;
    // constants
    double rho = 1025, g = 9.81, depthscale = 1;
    // initial conditions
    std::string zsinitfile;
    int hotstart = 0, hotstartfileno = 0, hotstartflow = 0;
    // wave boundary
    int wbctype = WBC_OFF, nonhspectrum = 0, m = 10, random = 1, nspr = 0, correctHm0 = 1, Tm01switch = 0,
        wbcversion = 3, nspectrumloc = 1, wbcScaleEnergy = 1, wbcRemoveStokes = 1, Sfold = 1, bclwonly = 0,
        cyclicdiradjust = 0;
    double taper = 100, nmax = 0.8, Hrms = -123, Tm01 = 10, Trep = -123, dir0 = 270, Tlong = -123, fcutoff = 0,
           trepfac = 0.01, sprdthr = 0, rt = 0, dtbc = 1, dthetaS_XB = 0, wbcEvarreduce = 1, wbcQvarreduce = 1;
    std::string bcfile;
    // flow boundary
    int front = 4, left = LR_NEUMANN, right = LR_NEUMANN, back = BACK_ABS_2D, ARC = 1, highcomp = 0,
        freewave = 0, tidetype = TT_VELOCITY, tideloc = 0, paulrevere = PR_LAND;
    double order = 2, epsi = -1, zs0 = 0;
    std::string zs0file;
    // discharges
    std::string disch_loc_file, disch_timeseries_file;
    int ndischarge = 0, ntdischarge = 0;
    // flow
    int bedfriction = BF_MANNING, smag = 1, dynamrough = 0, friction_turbulence = 0, friction_acceleration = 0,
        friction_infiltration = 0;
    double gamma_turb = 1, ci = 1;
    double bedfriccoef = 0.02, maxcf = 0.04, mincf = 0, nuh = 0.1, nuhfac = 0, nuhv = 1, droot = 0.5, dstem = 0.5;
    std::string bedfricfile;
    double wearth = 1.0 / 24.0, lat = 0;
    double rhoa = 1.25, Cd = 0.002, windv = 0, windth = 270;
    std::string windfile;
    std::string rainfallratefile; double rainfallrate = 0; int nrainfallrate = 0;
    // non-hydrostatic
    int solver = 0, solver_maxit = 30, nonhq3d = 0, nhbreaker = 1;
    double solver_acc = 0.005, solver_urelax = 0.92, kdmin = 0, Topt = 10, nhlay = 0.33, dispc = -1,
           breakviscfac = 1.5, maxbrsteep = 0.4, reformsteep = 0.1, secbrsteep = 0.2, breakvisclen = 1;
    // numerics
    double eps = 0.005, eps_sd = 0.5, umin = 0, hmin = 0, maxfacg = 10;
    int secorder = 1, oldhu = 0, oldhmin = 0;
    // output
    int nglobalvar = -1, npoints = 0, nrugauge = 0, npointvar = 0, nrugdepth = 1, nmeanvar = 0,
        outputformat = OUT_FORTRAN, outputdouble = 1, remdryoutput = 0, rotate = 1, writehotstart = 0, timings = 1;
    std::vector<double> rugdepth;
    std::vector<std::string> globalvars, pointvars, meanvars;
    struct Point { double x, y; std::string id; int type; };
    std::vector<Point> points;
    std::string ncfilename = "xboutput.nc", projection, tunits = "s";
    std::string drifterfile; int ndrifter = 0;
    std::string shipfile, veggiefile, veggiemapfile;
    // groundwater (groundwater.F90); bed composition por, D50
    int gwnonh = 0, gwfastsolve = 0, gwscheme = 0, gwheadmodel = 0, gwhorinfil = 0, ngd = 1, nd = 3;
    double kx = 0.0001, ky = 0.0001, kz = 0.0001, dwetlayer = 0.1, aquiferbot = -10, gw0 = 0, gwReturb = -123, por = 0.4;
    std::vector<double> D50, D90, D15;
    std::string aquiferbotfile, gw0file;
    // sediment transport (morphevolution.F90; sedtrans = 1). Option codes as paramsconst.F90.
    int q3d = 0, swrunup = 0, nz = 1;
    int form = 1, sws = 0, lws = 1, turb = 2, turbadv = 1, sus = 1, bed = 1, bulk = 0, bdslpeffmag = 1, bdslpeffini = 0,
        bdslpeffdir = 0, lwt = 0, fallvelred = 0, dilatancy = 0, bermslopetransport = 0, bermslopebed = 0,
        bermslopesus = 0, kmax = 1, phaselag = 0, sedfricfac = 1, streaming = 0, incldzdx = 0, inclrelweight = 0,
        slopecorr = -123, sourcesink = 0, oldTsmin = 0;
    double rhos = 2650, dzg1 = 0.1, dzg2 = 0.1, dzg3 = 0.1, cf_transport = 0.01, nonh_za = 0, turbcoef = 1,
           max_dzsdt = 10, ws_nonh = 0, facsl = 0.15, z0 = 0.006, smax = -1, bdslpeffdirfac = 1, reposeangle = 30,
           tsfac = 0.1, Tsmin = 0.5, facDc = 1, jetfac = 0, yturb = 1, facthr = 1, betad = 1, rheeA = 0.75,
           pormax = 0.5, bermslope = 0.1, bermslopefac = 15, bermslopedepth = 1, alfaD50 = 0.4, vonkar = 0.4,
           vicmol = 1e-6, sigfac = 1.3, deltar = 0.025, rwave = 2, Ctrans = 12, thetcr = -123, phit = 25, Arms = 10,
           fsed = 0.025, uprushfac = 1, backwashfac = 1, BRfac = 0, facua = 0, facSk = 0, facAs = 0, beta = 0.08,
           thetanum = 1, cmax = 0.1, dtlimTs = 5;
    std::vector<double> sedcal, ucrcal;
    // morphology
    int morfacopt = 1, struct_ = 0, nd_var = 2, fixedavaltime = 0, nsetbathy = 1;
    double morfac = 1, morstart = 0, morstop = 2000, wetslp = 0.15, dryslp = 1, lsgrad = 0, hswitch = 0, dzmax = 0.05,
           frac_dz = 0.7, split = 1.01, merge = 0.01, avaltime = 10, nTrepavaltime = 1;
    std::string ne_layer, setbathyfile;
    int vegnonlin = 0, vegcanflo = 0, veguntow = 1, porcanflow = 0;
    double Kp = 0.0001, Cm = 1.0;
    // GPU-specific keys
    double gpu_solver_acc = 1e-3; int gpu_maxit = 200, gpu_hotstart_keepuv = 1;
    // derived
    double theta0 = 0;   // s%theta0 from dir0
};

static void read_params(ParFile &F, Params &p) {
    p.useXBeachG = F.integer("useXBeachGSettings", 0);
    const int G = p.useXBeachG;   // XBeach-G preset: changes the defaults below as params.F90 does
    // check_instat_backward_compatibility: nonh = 1 and instat select the model and the boundary type
    int haveWM = -1, haveWBC = -1;
    if ((F.has("nonh") || G) && F.integer("nonh", G) == 1) haveWM = WM_NONH;
    if (F.has("instat")) {
        int instat = F.option("instat", {"stat", "bichrom", "ts_1", "ts_2", "jons", "swan", "vardens", "reuse",
                                         "ts_nonh", "off", "stat_table", "jons_table"}, 2,
                              {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "40", "41"});
        bool nh = haveWM == WM_NONH;
        switch (instat) {
        case 0: haveWM = nh ? WM_NONH : WM_STAT; haveWBC = WBC_PARAMS; break;
        case 8: haveWM = WM_NONH; haveWBC = WBC_TSNONH; break;
        case 9: haveWM = nh ? WM_NONH : WM_SURFBEAT; haveWBC = WBC_OFF; break;
        case 4: haveWM = nh ? WM_NONH : WM_SURFBEAT; haveWBC = WBC_PARAMETRIC; break;
        case 11: haveWM = nh ? WM_NONH : WM_SURFBEAT; haveWBC = WBC_JONSTABLE; break;
        case 5: haveWM = nh ? WM_NONH : WM_SURFBEAT; haveWBC = WBC_SWAN; break;
        case 6: haveWM = nh ? WM_NONH : WM_SURFBEAT; haveWBC = WBC_VARDENS; break;
        case 7: haveWM = nh ? WM_NONH : WM_SURFBEAT; haveWBC = WBC_REUSE; break;
        default: haveWM = WM_SURFBEAT; break;
        }
    }
    if (F.has("wavemodel")) p.wavemodel = F.option("wavemodel", {"stationary", "surfbeat", "nonh"}, 2, {"0", "1", "2"});
    else if (haveWM >= 0) p.wavemodel = haveWM;
    else fail("XBeach cannot run without a hydrodynamic type: set wavemodel = nonh");
    if (p.wavemodel != WM_NONH) fail("xbnh_gpu runs wavemodel = nonh only");

    p.cyclic = F.integer("cyclic", 0);
    p.swave = F.integer("swave", 0);
    p.lwave = F.integer("lwave", 1);
    p.flow = F.integer("flow", 1);
    p.sedtrans = F.integer("sedtrans", G);
    p.morphology = F.integer("morphology", p.sedtrans);
    p.avalanching = F.integer("avalanching", p.morphology);
    p.gwflow = F.integer("gwflow", G);
    p.q3d = F.integer("q3d", 0); p.swrunup = F.integer("swrunup", 0);
    if (p.swrunup) fail("swrunup = 1 needs short-wave forcing (swave = 1), not available in nonh");
    p.ships = F.integer("ships", 0);
    p.vegetation = F.integer("vegetation", 0);
    p.setbathy = F.integer("setbathy", 0);
    p.viscosity = F.integer("viscosity", 1);
    p.wind = F.integer("wind", 0);
    p.rainfall = F.integer("rainfall", 0);

    // grid
    p.gridform = F.option("gridform", {"xbeach", "delft3d"}, 0, {"0", "1"});
    p.hotstart = F.integer("hotstart", 0);
    p.hotstartflow = F.integer("hotstartflow", 0);
    if (p.hotstart == 1) p.hotstartfileno = F.integer("hotstartfileno", 0);
    p.xori = F.dbl("xori", 0); p.yori = F.dbl("yori", 0); p.alfa = F.dbl("alfa", 0);
    p.posdwn = F.dbl("posdwn", 1);
    if (p.gridform == 0) {
        p.nx = F.integer("nx", 50); p.ny = F.integer("ny", 2);
        if (!F.has("nx") || !F.has("ny")) fail("nx and ny are required");
        if (p.setbathy != 1 && p.hotstart != 1) { p.depfile = F.name("depfile"); if (p.depfile.empty()) fail("depfile is required"); }
        p.vardx = F.integer("vardx", 1);
        p.dx = F.dbl("dx", -1); p.dy = F.dbl("dy", -1);
        if (p.vardx == 0) { if (!(p.dx > 0) || !(p.dy > 0)) fail("vardx = 0 needs dx and dy"); }
        else { p.xfile = F.name("xfile"); p.yfile = F.name("yfile");
               if (p.xfile.empty()) fail("vardx = 1 needs xfile");
               if (p.ny > 0 && p.yfile.empty()) fail("vardx = 1 needs yfile"); }
    } else {
        p.depfile = F.name("depfile"); p.xyfile = F.name("xyfile");
        if (p.depfile.empty() || p.xyfile.empty()) fail("gridform = delft3d needs depfile and xyfile");
        p.dx = -1; p.dy = -1;
    }
    if (p.ny == 1 || p.ny == 2) fail("ny = 1 or 2 is not supported (use ny = 0 for 1D or ny >= 3)");

    // time
    if (G && p.ny > 0) fail("XBeach-G settings cannot be applied in 2DH models: set ny = 0");
    p.CFL = F.dbl("CFL", G ? 0.6 : (p.ny == 0 ? 0.95 : 0.7));
    p.tstop = F.dbl("tstop", 2000); if (!F.has("tstop")) fail("tstop is required");
    p.defuse = F.integer("defuse", 1);
    p.maxdtfac = F.dbl("maxdtfac", 500);
    p.rho = F.dbl("rho", 1025); p.g = F.dbl("g", 9.81); p.depthscale = F.dbl("depthscale", 1);
    p.zsinitfile = F.name("zsinitfile");

    // wave boundary
    if (F.has("wbctype"))
        p.wbctype = F.option("wbctype", {"params", "parametric", "swan", "vardens", "off", "jonstable", "reuse", "ts_1",
                                         "ts_2", "ts_nonh"}, 5, {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});
    else if (haveWBC >= 0) p.wbctype = haveWBC;
    else fail("XBeach cannot run without a type of boundary condition: set wbctype");
    if (p.wbctype == WBC_TS1 || p.wbctype == WBC_TS2) fail("ts_1 and ts_2 can only be used in surfbeat mode");
    bool spectral = p.wbctype == WBC_PARAMETRIC || p.wbctype == WBC_SWAN || p.wbctype == WBC_VARDENS ||
                    p.wbctype == WBC_JONSTABLE;
    if (spectral) { p.bcfile = F.name("bcfile"); if (p.bcfile.empty()) fail("wbctype needs bcfile"); }
    p.taper = F.dbl("taper", 100); p.nmax = F.dbl("nmax", 0.8);
    p.cyclicdiradjust = F.integer("cyclicdiradjust", p.cyclic);
    p.bclwonly = F.integer("bclwonly", 0); p.Sfold = F.integer("Sfold", 1);
    if (p.wbctype == WBC_PARAMS) {
        p.nonhspectrum = F.integer("nonhspectrum", 0);
        p.Hrms = F.dbl("Hrms", 1); p.Tm01 = F.dbl("Tm01", 10); p.Trep = F.dbl("Trep", p.Tm01);
        p.dir0 = F.dbl("dir0", 270); p.m = F.integer("m", 10);
    }
    if (F.has("Tlong")) p.Tlong = F.dbl("Tlong", 80);
    if (spectral) {
        p.nonhspectrum = 1;
        p.random = F.integer("random", 1); p.fcutoff = F.dbl("fcutoff", 0); p.nspr = F.integer("nspr", 0);
        p.trepfac = F.dbl("trepfac", 0.01); p.sprdthr = F.dbl("sprdthr", 0.0);
        p.correctHm0 = F.integer("correctHm0", 1); p.Tm01switch = F.integer("Tm01switch", 0);
        p.rt = F.dbl("rt", std::min(3600.0, p.tstop)); p.dtbc = F.dbl("dtbc", 1.0);
        if (p.wbctype == WBC_SWAN) p.dthetaS_XB = F.dbl("dthetaS_XB", 0);
        p.wbcversion = F.integer("wbcversion", 3);
        p.nspectrumloc = F.integer("nspectrumloc", 1);
        p.wbcEvarreduce = F.dbl("wbcEvarreduce", p.ny == 0 ? 0.3 : 1.0);
        p.wbcQvarreduce = F.dbl("wbcQvarreduce", 1.0);
        p.wbcScaleEnergy = F.integer("wbcScaleEnergy", 1); p.wbcRemoveStokes = F.integer("wbcRemoveStokes", 1);
    }
    if (p.wbctype == WBC_REUSE) p.nonhspectrum = 1;

    // flow boundaries
    F.option("front", {"abs_1d", "abs_2d", "wall", "wlevel", "nonh_1d", "waveflume"}, 4, {"0", "1", "2", "3", "4", "5"});
    p.front = 4;   // params.F90: front is set to nonh_1d whenever wavemodel = nonh
    std::vector<std::string> lr = {"neumann", "wall", "no_advec", "neumann_v", "abs_1d"};
    p.left = F.option("left", lr, 0, {"0", "1"}); p.right = F.option("right", lr, 0, {"0", "1"});
    p.back = F.option("back", {"wall", "abs_1d", "abs_2d", "wlevel"}, p.ny > 2 ? 2 : 1, {"0", "1", "2", "3"});
    if (p.back == 3) fail("wlevel is no longer supported (back)");
    if (p.back == BACK_ABS_2D && p.ny < 3) p.back = BACK_ABS_1D;
    p.ARC = F.integer("ARC", 1); p.order = F.dbl("order", 2); p.highcomp = F.integer("highcomp", 0);
    p.freewave = F.integer("freewave", 0); p.epsi = F.dbl("epsi", -1);
    p.tidetype = F.option("tidetype", {"instant", "velocity", "hybrid"}, 1);
    p.tideloc = F.integer("tideloc", F.has("zs0") ? 0 : 2);
    if (p.tideloc > 0) {
        if (p.tideloc == 2) p.paulrevere = F.option("paulrevere", {"land", "sea"}, 0, {"0", "1"});
        p.zs0file = F.name("zs0file");
        if (p.zs0file.empty()) fail("tideloc > 0 needs zs0file (or set zs0)");
        if (p.tideloc == 3 || p.tideloc > 4) fail("tideloc must be 0, 1, 2 or 4");
    } else p.zs0 = F.dbl("zs0", 0);
    if (p.tidetype == TT_HYBRID) fail("tidetype = hybrid needs short-wave forcing (swave = 1), not available in nonh");

    // discharges
    p.disch_loc_file = F.name("disch_loc_file"); p.disch_timeseries_file = F.name("disch_timeseries_file");
    auto flen = [](const std::string &fn) { if (fn.empty()) return 0; std::ifstream f(fn); std::string l; int n = 0;
                                            while (std::getline(f, l)) if (trim(l).size()) n++; return n; };
    p.ndischarge = F.integer("ndischarge", flen(p.disch_loc_file));
    p.ntdischarge = F.integer("ntdischarge", flen(p.disch_timeseries_file));

    // friction, viscosity, Coriolis, wind, rainfall
    p.bedfriction = F.option("bedfriction", {"chezy", "cf", "white-colebrook", "manning", "white-colebrook-grainsize"}, G ? 4 : 3);
    if (!F.has("bedfriction") && F.has("bedfricfile")) fail("bedfricfile without bedfriction is not allowed");
    if (p.bedfriction != BF_WCG) {
        p.bedfricfile = F.name("bedfricfile");
        if (p.bedfricfile.empty()) {
            const double defs[] = {55, 3e-3, 0.01, 0.02};
            p.bedfriccoef = F.dbl("bedfriccoef", defs[p.bedfriction]);
        }
    } else p.bedfriccoef = -999;
    p.dynamrough = F.integer("dynamicroughness", 0);
    p.droot = F.dbl("droot", 0.5); p.dstem = F.dbl("dstem", 0.5);
    if (G) { p.maxcf = 0.1; p.mincf = 0; } else { p.maxcf = F.dbl("maxcf", 0.04); p.mincf = F.dbl("mincf", 0); }
    p.nuh = F.dbl("nuh", 0.1); p.nuhfac = F.dbl("nuhfac", 0); p.nuhv = F.dbl("nuhv", 1); p.smag = F.integer("smag", 1);
    if (p.gwflow == 1) p.friction_infiltration = F.integer("friction_infiltration", G);
    p.friction_turbulence = F.integer("friction_turbulence", 0);
    p.friction_acceleration = F.option("friction_acceleration", {"none", "mccall", "nielsen"}, G ? 1 : 0);
    if (p.friction_turbulence == 1) p.gamma_turb = F.dbl("gamma_turb", 1.0);
    if (p.friction_acceleration == 1) p.ci = F.dbl("ci", 1.0);
    else if (p.friction_acceleration == 2) p.phit = F.dbl("phit", 25.0);
    p.wearth = F.dbl("wearth", 1.0 / 24.0); p.lat = F.dbl("lat", 0);
    if (p.wind == 1) {
        p.rhoa = F.dbl("rhoa", 1.25); p.Cd = F.dbl("Cd", 0.002); p.windfile = F.name("windfile");
        if (p.windfile.empty()) { p.windv = F.dbl("windv", 0); p.windth = F.dbl("windth", 270); }
    } else { p.windv = 0; p.windth = 0; p.windfile = ""; }
    if (p.rainfall == 1) {
        p.rainfallratefile = F.name("rainfallratefile");
        if (p.rainfallratefile.empty()) p.rainfallrate = F.dbl("rainfallrate", 0);
        else p.nrainfallrate = F.integer("nrainfallrate", 1);
    }

    // non-hydrostatic
    p.solver = F.option("solver", {"sip", "tridiag"}, p.ny > 2 ? 0 : 1, {"1", "2"});
    if (p.solver == 0) { p.solver_maxit = F.integer("solver_maxit", 30); p.solver_acc = F.dbl("solver_acc", 0.005);
                         p.solver_urelax = F.dbl("solver_urelax", 0.92); }
    p.kdmin = F.dbl("kdmin", 0); p.Topt = F.dbl("Topt", 10);
    p.nonhq3d = F.integer("nonhq3d", 0);
    p.nhbreaker = F.integer("nhbreaker", 1);
    if (p.nonhq3d == 1) p.nhlay = F.dbl("nhlay", 0.33); else p.dispc = F.dbl("dispc", -1);
    if (p.nhbreaker == 1) {
        p.breakviscfac = F.dbl("breakviscfac", 1.5); p.maxbrsteep = F.dbl("maxbrsteep", 0.4);
        p.reformsteep = F.dbl("reformsteep", 0.25 * p.maxbrsteep);
    } else if (p.nhbreaker == 2 || p.nhbreaker == 3) {
        if (!G) fail("nhbreaker = 2 and 3 need useXBeachGSettings = 1 (XBeach accepts 0 and 1 otherwise)");
        p.breakvisclen = F.dbl("breakvisclen", 1.0); p.maxbrsteep = F.dbl("maxbrsteep", p.nhbreaker == 2 ? 0.4 : 0.6);
        p.secbrsteep = F.dbl("secbrsteep", 0.5 * p.maxbrsteep);
    } else if (p.nhbreaker != 0) fail("nhbreaker must be 0, 1, 2 or 3");

    // numerics
    p.oldhmin = F.integer("oldhmin", 0);
    p.eps = F.dbl("eps", 0.005); p.eps_sd = F.dbl("eps_sd", 0.5); p.umin = F.dbl("umin", 0);
    p.hmin = p.oldhmin == 1 ? F.dbl("hmin", 0.2) : 0;
    p.secorder = F.integer("secorder", 1); p.oldhu = F.integer("oldhu", 0); p.maxfacg = 0;

    // output
    p.timings = F.integer("timings", 1);
    if (F.has("tunits")) p.tunits = F.name("tunits");
    p.tstart = F.dbl("tstart", 0);
    p.tint = F.dbl("tint", 900);
    p.tsglobal = F.name("tsglobal"); if (p.tsglobal.empty()) p.tintg = F.dbl("tintg", p.tint);
    p.tspoints = F.name("tspoints"); if (p.tspoints.empty()) p.tintp = F.dbl("tintp", 1);
    p.tsmean = F.name("tsmean"); if (p.tsmean.empty()) p.tintm = F.dbl("tintm", p.tstop - p.tstart);
    p.writehotstart = F.integer("writehotstart", 0);
    if (p.writehotstart == 1) { p.tshotstart = F.name("tshotstart"); if (p.tshotstart.empty()) p.tinth = F.dbl("tinth", p.tstop); }
    p.nglobalvar = F.integer("nglobalvar", -1);
    p.npoints = F.integer("npoints", 0); p.nrugauge = F.integer("nrugauge", 0); p.npointvar = F.integer("npointvar", 0);
    p.nrugdepth = F.integer("nrugdepth", 1);
    p.nmeanvar = F.integer("nmeanvar", 0);
#ifdef XB_NETCDF
    p.outputformat = F.option("outputformat", {"fortran", "netcdf", "debug"}, 1);
#else
    p.outputformat = F.option("outputformat", {"fortran", "netcdf", "debug"}, 0);
    if (p.outputformat != OUT_FORTRAN) fail("this xbnh_gpu build has no netCDF support: set outputformat = fortran or rebuild with NETCDF=1");
#endif
    if (p.outputformat != OUT_FORTRAN) {
        p.outputdouble = F.option("outputprecision", {"single", "double"}, 1);
        p.ncfilename = F.name("ncfilename", "xboutput.nc");
    }
    p.remdryoutput = F.integer("remdryoutput", (p.outputformat == OUT_NETCDF && !G) ? 1 : 0);
    p.projection = F.name("projection"); p.rotate = F.integer("rotate", 1);
    if (F.has("drifterfile")) { p.drifterfile = F.name("drifterfile");
        std::ifstream f(p.drifterfile); if (!f) fail("cannot open drifterfile " + p.drifterfile);
        std::string l; int n = 0; while (std::getline(f, l)) if (trim(l).size()) n++;
        p.ndrifter = F.integer("ndrifter", n); }
    if (p.gwflow == 1) {   // params.F90 groundwater parameters
        p.kx = F.dbl("kx", G ? 0.01 : 0.0001); p.ky = F.dbl("ky", p.kx); p.kz = F.dbl("kz", p.kx);
        p.dwetlayer = F.dbl("dwetlayer", 0.1);
        p.aquiferbotfile = F.name("aquiferbotfile");
        if (p.aquiferbotfile.empty()) p.aquiferbot = F.dbl("aquiferbot", -10.0);
        p.gw0file = F.name("gw0file");
        if (p.gw0file.empty()) p.gw0 = F.dbl("gw0", 0.0);
        p.gwnonh = F.integer("gwnonh", G);
        if (p.gwnonh == 1 && p.ny > 2) p.gwfastsolve = F.integer("gwfastsolve", 0);
        p.gwscheme = F.option("gwscheme", {"laminar", "turbulent"}, G ? 1 : 0, {"darcy", "modflow"});
        if (p.gwscheme == 1) p.gwReturb = F.dbl("gwReturb", 100.0);
        p.gwheadmodel = F.option("gwheadmodel", {"parabolic", "exponential"}, 0);
        p.gwhorinfil = F.integer("gwhorinfil", 0);
        if (p.gwfastsolve == 1) fail("gwfastsolve = 1 is not supported (XBeach r6167 fills its matrix with an uninitialised row index)");
    }
    if (p.ships == 1) p.shipfile = F.name("shipfile");
    if (p.vegetation == 1) {   // params.F90 re-reads Trep here with a default of 1 s
        p.veggiefile = F.name("veggiefile"); p.veggiemapfile = F.name("veggiemapfile");
        if (p.veggiefile.empty() || p.veggiemapfile.empty()) fail("vegetation = 1 needs veggiefile and veggiemapfile");
        p.Trep = F.dbl("Trep", 1.0);
        p.vegnonlin = F.integer("vegnonlin", 0); p.vegcanflo = F.integer("vegcanflo", 0);
        p.veguntow = F.integer("veguntow", 1); p.porcanflow = F.integer("porcanflow", 0);
        if (p.porcanflow == 1) { p.Kp = F.dbl("Kp", 0.0001); p.Cm = F.dbl("Cm", 1.0); }
    }
    // sediment transport parameters (params.F90, wavemodel = nonh branches)
    p.beta = F.dbl("beta", 0.08);
    if (p.sedtrans == 1) {
        p.form = F.option("form", {"soulsby_vanrijn", "vanthiel_vanrijn", "vanrijn1993", "nielsen2006", "mccall_vanrijn",
                                   "wilcock_crow", "engelund_fredsoe", "mpm", "wong_parker", "fl_vb", "fredsoe_deigaard",
                                   "intrasedtr"}, G ? 4 : 1, {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"});
        p.sws = 0; p.lws = 1; p.BRfac = 0; p.facua = 0; p.facSk = 0; p.facAs = 0; p.turb = 2; p.turbadv = 1;
        bool gravel = p.form >= 3 && p.form <= 10;
        if (gravel) {
            if (p.form == 3) {
                p.Ctrans = F.dbl("Ctrans", 12.0); p.thetcr = F.dbl("thetcr", 0.05); p.phaselag = F.integer("phaselag", 0);
                if (p.phaselag == 1) p.phit = F.dbl("phit", 25.0);
                p.sedfricfac = F.option("sedfricfac", {"constant", "flowfricfac", "nielsen", "swart", "wilson"}, 0,
                                        {"0", "1", "2", "3", "4"});
                if (p.sedfricfac == 3) p.Arms = F.dbl("Arms", 10.0);
                if (p.sedfricfac == 0) p.fsed = F.dbl("fsed", 0.025);
                p.streaming = F.integer("streaming", 0);
            } else {
                p.thetcr = F.dbl("thetcr", -10.0); p.sedfricfac = 1;
                p.uprushfac = F.dbl("uprushfac", 1.0); p.backwashfac = F.dbl("backwashfac", 1.0);
                p.incldzdx = F.integer("incldzdx", 0);
                p.inclrelweight = p.gwflow == 1 ? F.integer("inclrelweight", 1) : 0;
            }
            p.slopecorr = F.option("slopecorr", {"none", "nielsen", "fredsoe_deigaard"}, 2, {"0", "1", "2"});
            p.sus = F.integer("sus", 0); p.bed = F.integer("bed", 1); p.bulk = F.integer("bulk", 0);
            p.facsl = p.bulk == 0 ? 0.0 : F.dbl("facsl", 0.0);
            p.bdslpeffmag = 0; p.bdslpeffini = 0; p.bdslpeffdir = 0; p.smax = -1;
        } else {
            p.cf_transport = F.dbl("cf_transport", 0.01); p.nonh_za = F.dbl("nonh_za", 0.0);
            p.turbcoef = F.dbl("turbcoef", 1.0); p.max_dzsdt = F.dbl("max_dzsdt", 10.0); p.ws_nonh = F.dbl("ws_nonh", 0.0);
            p.sus = F.integer("sus", 1); p.bed = F.integer("bed", 1); p.bulk = F.integer("bulk", 0);
            p.facsl = F.dbl("facsl", 0.15); p.z0 = F.dbl("z0", 0.006); p.smax = F.dbl("smax", -1.0);
            p.bdslpeffmag = F.option("bdslpeffmag", {"none", "roelvink_total", "roelvink_bed", "soulsby_total", "soulsby_bed"}, 1,
                                     {"0", "1", "2", "3", "4"});
            p.bdslpeffini = F.option("bdslpeffini", {"none", "total", "bed"}, 0, {"0", "1", "2"});
            p.bdslpeffdir = F.option("bdslpeffdir", {"none", "talmon"}, 0, {"0", "1"});
            if (p.bdslpeffdir > 0) p.bdslpeffdirfac = F.dbl("bdslpeffdirfac", 1.0);
        }
        p.reposeangle = F.dbl("reposeangle", G ? 35.0 : 30.0); p.tsfac = F.dbl("tsfac", 0.1); p.Tsmin = F.dbl("Tsmin", 0.5);
        p.facDc = F.dbl("facDc", 1.0); p.jetfac = F.dbl("jetfac", 0.0); p.lwt = F.integer("lwt", 0);
        if (p.lwt == 1 && (p.form == 3 || p.form == 11)) { p.yturb = F.dbl("yturb", 1.0); p.facthr = F.dbl("facthr", 1.0); }
        p.betad = F.dbl("betad", 1.0); p.fallvelred = F.integer("fallvelred", 0); p.dilatancy = F.integer("dilatancy", 0);
        if (p.dilatancy == 1) { p.rheeA = F.dbl("rheeA", 0.75); p.pormax = F.dbl("pormax", 0.5); }
        p.bermslopetransport = F.integer("bermslopetransport", 0);
        if (p.bermslopetransport == 1) {
            p.bermslopebed = F.integer("bermslopebed", 1); p.bermslopesus = F.integer("bermslopesus", 1);
            p.bermslope = F.dbl("bermslope", 0.1); p.bermslopefac = F.dbl("bermslopefac", 15.0);
            p.bermslopedepth = F.dbl("bermslopedepth", 1.0);
        } else { p.bermslopebed = 0; p.bermslopesus = 0; }
        p.alfaD50 = F.dbl("alfaD50", 0.4);
    }
    p.nz = F.integer("nz", 1);
    if (p.q3d == 1 || (p.sedtrans == 1 && p.form == 2)) {
        p.vonkar = F.dbl("vonkar", 0.4); p.vicmol = F.dbl("vicmol", 0.000001); p.kmax = F.integer("kmax", 100);
        p.nz = p.kmax; p.sigfac = F.dbl("sigfac", 1.3); p.deltar = F.dbl("deltar", 0.025); p.rwave = F.dbl("rwave", 2.0);
    } else p.kmax = 1;
    // bed composition (read in every run)
    p.ngd = F.integer("ngd", 1); p.nd = F.integer("nd", 3); p.por = F.dbl("por", 0.4);
    if (p.dilatancy == 1) p.D15 = F.dblvec("D15", p.ngd, 0.00015);
    p.D50 = F.dblvec("D50", p.ngd, G ? 0.010 : 0.0002); p.D90 = F.dblvec("D90", p.ngd, G ? 0.015 : 0.0003);
    if (p.sedtrans == 1) {
        p.rhos = F.dbl("rhos", 2650.0);
        p.dzg1 = F.dbl("dzg", 0.1); p.dzg1 = F.dbl("dzg1", p.dzg1); p.dzg2 = F.dbl("dzg2", p.dzg1); p.dzg3 = F.dbl("dzg3", p.dzg1);
        p.sedcal = F.dblvec("sedcal", p.ngd, 1.0); p.ucrcal = F.dblvec("ucrcal", p.ngd, 1.0);
    } else { p.sedcal.assign(p.ngd, 1.0); p.ucrcal.assign(p.ngd, 1.0); }
    if (p.morphology == 1) {
        p.morfac = F.dbl("morfac", 1.0); p.morfacopt = F.integer("morfacopt", 1);
        p.morstart = F.dbl("morstart", 0.0); p.morstop = F.dbl("morstop", p.tstop);
        if (G) { p.wetslp = F.dbl("wetslp", tan(p.reposeangle / 180.0 * 4.0 * atan(1.0))); p.dryslp = F.dbl("dryslp", p.wetslp); }
        else { p.wetslp = F.dbl("wetslp", 0.15); p.dryslp = F.dbl("dryslp", 1.0); }
        p.lsgrad = p.ny == 0 ? F.dbl("lsgrad", 0.0) : 0.0;
        p.hswitch = F.dbl("hswitch", 0.0); p.dzmax = F.dbl("dzmax", 0.05); p.struct_ = F.integer("struct", 0);
        if (p.struct_ == 1 && p.hotstart != 1) { p.ne_layer = F.name("ne_layer"); if (p.ne_layer.empty()) fail("struct = 1 needs ne_layer"); }
    } else { p.morstop = p.tstop; p.morfac = -123; }   // params.def default when morphology = 0
    // sediment transport and bed update numerics
    if (p.sedtrans == 1) {
        p.thetanum = F.dbl("thetanum", 1.0); p.sourcesink = F.integer("sourcesink", 0); p.cmax = F.dbl("cmax", 0.1);
        p.oldTsmin = F.integer("oldTsmin", 0);
        if (p.oldTsmin == 0) p.dtlimTs = F.dbl("dtlimTs", 5.0);
    }
    if (p.morphology == 1) {
        p.frac_dz = F.dbl("frac_dz", 0.7); p.nd_var = F.integer("nd_var", 2); p.split = F.dbl("split", 1.01);
        p.merge = F.dbl("merge", 0.01); p.fixedavaltime = F.integer("fixedavaltime", 0);
        if (p.fixedavaltime == 1) p.avaltime = F.dbl("avaltime", 10.0); else p.nTrepavaltime = F.dbl("nTrepavaltime", 1.0);
    }
    if (p.setbathy == 1) {
        p.nsetbathy = F.integer("nsetbathy", 1); p.setbathyfile = F.name("setbathyfile");
        if (p.setbathyfile.empty()) fail("setbathy = 1 needs setbathyfile");
    }

    // global, point and mean variable lists
    if (p.nglobalvar == -1) {
        p.globalvars = {"H", "zs", "zs0", "zb", "hh", "u", "v", "ue", "ve", "urms", "Fx", "Fy", "ccg", "ceqsg",
                        "ceqbg", "Susg", "Svsg", "E", "R", "D", "DR"};
        p.nglobalvar = 21;
    } else if (p.nglobalvar == 999) fail("nglobalvar = 999 (all variables) is not supported; list the variables");
    else for (auto &l : F.list_after("nglobalvar", p.nglobalvar)) { std::istringstream is(l); std::string w; is >> w; p.globalvars.push_back(w); }
    auto readpts = [&](const char *key, int n, int type, const char *prefix) {
        int q = 0;
        for (auto &l : F.list_after(key, n)) {
            q++;
            if (l.find('#') != std::string::npos) fail("old '#' point output syntax is not supported");
            std::string ll = l; for (auto &ch : ll) if (ch == ',') ch = ' ';
            std::istringstream is(ll); Params::Point pt{}; pt.type = type;
            if (!(is >> pt.x >> pt.y)) fail(std::string("cannot read ") + key + " line: " + l);
            if (!(is >> pt.id)) { char b[32]; snprintf(b, sizeof b, "%s%03d", prefix, q); pt.id = b; }
            p.points.push_back(pt);
        }
    };
    if (p.npoints > 0) {
        readpts("npoints", p.npoints, 0, "point");
        if (!F.has("npointvar") && p.nrugauge <= 0) fail("npoints > 0 needs npointvar");
        for (auto &l : F.list_after("npointvar", p.npointvar)) { std::istringstream is(l); std::string w; is >> w; p.pointvars.push_back(w); }
    }
    if (p.nrugauge > 0) {
        readpts("nrugauge", p.nrugauge, 1, "rugau");
        for (const char *v : {"xz", "yz", "zs"})
            if (std::find(p.pointvars.begin(), p.pointvars.end(), v) == p.pointvars.end()) { p.pointvars.push_back(v); p.npointvar++; }
    }
    for (size_t a = 0; a < p.points.size(); a++)
        for (size_t b = a + 1; b < p.points.size(); b++)
            if (p.points[a].id == p.points[b].id) fail("output points " + p.points[a].id + " defined twice");
    p.rugdepth.assign(10, 0);
    if (p.nrugauge > 0) {
        auto v = F.find("rugdepth");
        std::vector<double> rd;
        if (v) { std::string s = *v; for (auto &ch : s) if (ch == ',') ch = ' '; std::istringstream is(s); double x; while (is >> x) rd.push_back(x); }
        for (int q = 0; q < 10; q++) p.rugdepth[q] = q < (int)rd.size() && q < p.nrugdepth ? rd[q] : 0;
        for (int q = p.nrugdepth; q < 10; q++) p.rugdepth[q] = (1.0 + 2.220446049250313e-16) * p.eps;
        for (int q = 0; q < 10; q++) if (p.rugdepth[q] <= p.eps) p.rugdepth[q] = (1.0 + 2.220446049250313e-16) * p.eps;
    }
    if (p.nmeanvar > 0)
        for (auto &l : F.list_after("nmeanvar", p.nmeanvar)) { std::istringstream is(l); std::string w; is >> w; p.meanvars.push_back(w); }

    // GPU keys
    p.gpu_solver_acc = F.dbl("gpu_solver_acc", 1e-3);
    p.gpu_maxit = F.integer("gpu_maxit", 200);
    p.gpu_hotstart_keepuv = F.integer("gpu_hotstart_keepuv", 1);

    // post-processing (params.F90 after reading)
    p.maxfacg = F.dbl("maxfacg", 10);
    if (p.depthscale != 1) { p.eps /= p.depthscale; p.hmin /= p.depthscale; }
    if (p.posdwn < 0.1) p.posdwn = -1;
    if (p.sedtrans == 0 && p.morphology == 1) fail("morphology cannot be computed without sediment transport (sedtrans = 1)");
    if (p.morphology == 0) p.avalanching = 0;
    if (p.setbathy == 1) { p.morphology = 0; p.avalanching = 0; }
    p.swave = 0;
    if (p.sedtrans == 1 && p.nz > 1 && p.form != 2) fail("the quasi-3D velocity profile (nz > 1) needs short-wave forcing, not available in nonh");
    if (p.smax < 0) p.smax = std::numeric_limits<double>::max() * 1.0e-20;
    if (p.morfacopt == 1) {   // times in hydrodynamic seconds
        double mf = std::max(p.morfac, 1.0);
        p.tstart /= mf; p.tint /= mf; p.tintg /= mf; p.tintp /= mf; p.tintm /= mf; p.tinth /= mf;
        p.tstop /= mf; p.morstart /= mf; p.morstop /= mf; p.rt /= mf;
    }
    if (p.wbctype == WBC_PARAMS && p.Tlong != -123) fail("bichromatic waves can only be used in surfbeat mode");
    if (p.cyclic == 1 && p.ny < 5) fail("cyclic = 1 needs ny >= 5");
    p.taper = std::max(p.taper, 1e-6);
    p.lat = p.lat * M_PI / 180.0;
    p.wearth = M_PI * p.wearth / 1800.0;
    p.secorder = 1;   // always on with wavemodel = nonh
    if (p.ny > 2 && p.solver == 1) fail("tri-diagonal solver cannot be used if ny > 2");
    if (p.ny == 0 && p.solver == 0) fail("SIP solver cannot be used if ny == 0");
    if (p.nonhspectrum == 1) p.wbcversion = 3;
    if (p.nhbreaker == 1) {
        if (p.reformsteep > 0.95 * p.maxbrsteep) p.reformsteep = 0.95 * p.maxbrsteep;
        else if (p.reformsteep < 0) p.reformsteep = 0;
    }
    if (p.tintm > p.tstop - p.tstart && p.nmeanvar > 0) p.tintm = p.tstop - p.tstart;
    // theta0 (grid_bathy, swave = 0 branch)
    double th0 = 1.5 * M_PI - p.dir0 * atan(1.0) / 45.0;
    while (th0 < -M_PI) th0 += 2 * M_PI;
    while (th0 > M_PI) th0 -= 2 * M_PI;
    p.theta0 = th0;
    // wave_init: Trep = 10 for spectral, reuse and ts_nonh boundaries until a boundary file sets it
    if (spectral || p.wbctype == WBC_REUSE || p.wbctype == WBC_TSNONH) p.Trep = 10;
}
