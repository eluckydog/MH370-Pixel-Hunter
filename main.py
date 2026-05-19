#!/usr/bin/env python3
"""
MH370 Pixel Hunter — 像素猎手
=======================================
Unified CLI Entry Point

Integrates four analysis paths into one tool:

  skyprint       — Satellite contrail/candidate screening pipeline
  intersect      — Multi-constraint intersection (debris+arc+fuel+radar)
  detectability  — MTSAT-2 VIS debris field detectability simulation
  flight         — BTO/BFO path analysis, Monte Carlo simulation

Usage:
  python main.py skyprint --help
  python main.py intersect --help
  python main.py detectability --help
  python main.py flight --help
"""

import sys
import os

# Ensure submodules are importable
SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
for sub in ('skyprint', 'modeling', 'flight'):
    sys.path.insert(0, os.path.join(SKILL_DIR, sub))

import argparse


def cmd_skyprint(args):
    """Run SkyPrint contrail screening pipeline."""
    from skyprint.skyprint_agent import main as skyprint_main
    # Forward remaining args
    sys.argv = ['skyprint'] + args.remainder
    skyprint_main()


def cmd_intersect(args):
    """Run multi-constraint intersection model (debris + arc + fuel + radar)."""
    sys.path.insert(0, os.path.join(SKILL_DIR, 'modeling'))
    from multi_constraint_intersection import (
        run_intersection_model, plot_intersection, save_results
    )
    result = run_intersection_model(
        n_samples=args.samples,
        debris_points=args.debris,
        arc_center=(args.arc_lon, args.arc_lat),
        arc_radius=args.arc_radius,
    )
    plot_intersection(result, output=args.output)
    if args.output:
        save_results(result, args.output)
    return result


def cmd_detectability(args):
    """Run MTSAT-2 debris field detectability simulation."""
    sys.path.insert(0, os.path.join(SKILL_DIR, 'modeling'))
    from mtsat2_detectability import run_detectability_simulation
    result = run_detectability_simulation(
        debris_radius_km=args.debris_radius,
        target_lat=args.lat,
        target_lon=args.lon,
        output_dir=args.output,
    )
    return result


def cmd_flight(args):
    """Run MH370 flight path analysis."""
    sys.path.insert(0, os.path.join(SKILL_DIR, 'flight'))
    if args.analysis == 'bfo':
        from flight.bfo_analysis import analyze_bfo
        analyze_bfo()
    elif args.analysis == 'mc':
        from flight.flight_path_monte_carlo import run_mc_simulation
        run_mc_simulation(n_samples=args.samples)
    elif args.analysis == 'currents':
        from flight.oscar_currents import run_backtrack
        run_backtrack(debris_points=args.debris)
    else:
        print(f"Unknown flight analysis: {args.analysis}")
        print("Options: bfo, mc, currents")


def main():
    parser = argparse.ArgumentParser(
        description='MH370 Pixel Hunter — 像素猎手',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py skyprint --bbox "31S,38S,92E,104E" --date 2014-03-08 --output review/
  python main.py data --date 2014-03-08 --bbox 92,-40,104,-25 --full
  python main.py intersect --samples 50000 --output intersection.png
  python main.py detectability --lat -35.5 --lon 95.5 --output detect/
  python main.py flight bfo
  python main.py flight mc --samples 100000
  python main.py flight currents --debris "-21.2,55.5;-19.7,63.4"
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Analysis module')

    # --- skyprint ---
    sp = subparsers.add_parser('skyprint', help='Satellite contrail screening')
    sp.add_argument('--bbox', type=str, required=True,
                    help='Bounding box, e.g. "31S,38S,92E,104E"')
    sp.add_argument('--date', type=str, required=True,
                    help='Start date YYYY-MM-DD')
    sp.add_argument('--window', type=int, default=1,
                    help='Search window in days')
    sp.add_argument('--layer', type=str, default='MODIS_Terra_TrueColor',
                    help='GIBS layer name')
    sp.add_argument('--output', type=str, default='review_package',
                    help='Output directory')
    sp.add_argument('--no-texture', action='store_true')
    sp.add_argument('--no-parallel', action='store_true')
    sp.add_argument('--no-temporal', action='store_true')

    # --- intersect ---
    ip = subparsers.add_parser('intersect', help='Multi-constraint intersection')
    ip.add_argument('--samples', type=int, default=50000,
                    help='Monte Carlo samples')
    ip.add_argument('--debris', type=str, default=None,
                    help='Semicolon-separated lat,lon pairs')
    ip.add_argument('--arc-lon', type=float, default=64.5,
                    help='Inmarsat sub-satellite longitude')
    ip.add_argument('--arc-lat', type=float, default=0.0,
                    help='Inmarsat sub-satellite latitude')
    ip.add_argument('--arc-radius', type=float, default=44.5,
                    help='7th arc radius in degrees')
    ip.add_argument('--output', type=str, default='intersection.png',
                    help='Output image path')

    # --- detectability ---
    # --- data --- (NEW: satellite data availability check)
    dcp = subparsers.add_parser('data', help='Check satellite data coverage')
    dcp.add_argument('--date', type=str, default='2014-03-08',
                     help='Date YYYY-MM-DD (default: MH370 disappearance)')
    dcp.add_argument('--bbox', type=str, default=None,
                     help='Bounding box min_lon,min_lat,max_lon,max_lat')
    dcp.add_argument('--full', action='store_true',
                     help='Full probe (all sources, slower)')

    # --- detectability ---
    dp = subparsers.add_parser('detectability', help='MTSAT-2 detectability sim')
    dp.add_argument('--debris-radius', type=float, default=0.15,
                    help='Debris object radius (km)')
    dp.add_argument('--lat', type=float, default=-35.5,
                    help='Target latitude')
    dp.add_argument('--lon', type=float, default=95.5,
                    help='Target longitude')
    dp.add_argument('--output', type=str, default='detect_output',
                    help='Output directory')

    # --- flight ---
    fp = subparsers.add_parser('flight', help='Flight path analysis')
    fp.add_argument('analysis', type=str, nargs='?', default='mc',
                    choices=['bfo', 'mc', 'currents'],
                    help='Analysis type')
    fp.add_argument('--samples', type=int, default=100000,
                    help='MC samples (mc only)')
    fp.add_argument('--debris', type=str, default=None,
                    help='Debris points for backtracking')

    args = parser.parse_args()

    if args.command == 'skyprint':
        # Convert argparse to sys.argv for skyprint_agent.main()
        sys.argv = ['mh370'] + [x for a in [
            '--bbox', args.bbox, '--start-date', args.date,
            '--window', str(args.window), '--layer', args.layer,
            '--output', args.output
        ] + (['--no-texture'] if args.no_texture else [])
          + (['--no-parallel'] if args.no_parallel else [])
          + (['--no-temporal'] if args.no_temporal else [])
          for x in a]
        cmd_skyprint(args)
    elif args.command == 'intersect':
        cmd_intersect(args)
    elif args.command == 'detectability':
        cmd_detectability(args)
    elif args.command == 'data':
        from data_sources import check_data_availability, search_modis_in_area
        from datetime import datetime
        
        if args.bbox:
            parts = [float(x) for x in args.bbox.replace(',', ' ').split()]
            bbox = (parts[0], parts[1], parts[2], parts[3]) if len(parts) == 4 else None
        else:
            # Default: SIO search area
            bbox = (92, -40, 104, -25)
        
        print(f"\n{'='*60}")
        print(f"MH370 Pixel Hunter — Data Coverage Check")
        print(f"Date: {args.date} | Bbox: {bbox}")
        print(f"{'='*60}\n")
        
        results = check_data_availability(args.date, bbox)
        
        print("\nResults:")
        for src, status in results.items():
            print(f"  {src:25s}: {status}")
        
        if args.full:
            # Extra searches
            for coll, label in [('MYD021KM', 'MODIS_Aqua_L1B'),
                                ('VNP02IMG', 'VIIRS_SDR'),
                                ('VNP02DNB', 'VIIRS_DNB'),
                                ('MOD06_L2', 'MODIS_Cloud'),
                                ('MYD06_L2', 'MODIS_Aqua_Cloud')]:
                g = search_modis_in_area(bbox, args.date, coll)
                print(f"  CMR {label:25s}: {len(g)} granules")
    elif args.command == 'flight':
        cmd_flight(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
